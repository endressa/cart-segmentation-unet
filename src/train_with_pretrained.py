import os
import csv
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from torch.utils.data import ConcatDataset, Subset
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG (paths are what you showed in your last message)
# ────────────────────────────────────────────────────────────────────────────────
# ── DATA ROOTS ────────────────────────────────────────────────────────────────
CLEAN_IMG_ROOT  = Path("~/sarah/background_segmentation/dataset/images_cleaned").expanduser()
CLEAN_MASK_ROOT = Path("~/sarah/background_segmentation/dataset/final_masks_cleaned").expanduser()

PSEUDO_IMG_ROOT  = Path("~/sarah/background_segmentation/dataset/images_pseudo").expanduser()
PSEUDO_MASK_ROOT = Path("~/sarah/background_segmentation/dataset/pseudo_masks").expanduser()

# How much to trust pseudo labels compared to clean (0.3–0.7 is typical)
PSEUDO_LOSS_WEIGHT = 0.5
# where to save the model + metrics
CHECKPOINT_PATH = Path("~/sarah/background_segmentation/checkpoints_pretrained/pseudo_model_12.pth").expanduser()
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


# training size (try 512; lower to 384 if VRAM is tight)
IMG_SIZE: Tuple[int, int] = (512, 512)
SIDE_PADDING_RATIO = 0.1  # must match what you used when generating masks

BATCH_SIZE = 6          # adjust per VRAM (try 4–8 at 512)
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7            # early stopping on Val Dice (no improvement)



def letterbox_image_with_side_padding(image, padding_color=(0, 0, 0), side_padding_ratio=SIDE_PADDING_RATIO):
    """
    Applies letterboxing to an image by:
    - Adding a specified % of horizontal padding (left & right),
    - Making the result square without resizing the original content.

    Parameters:
    - image (numpy.ndarray): The input image.
    - padding_color (tuple): Color of the padding as (B, G, R).
    - side_padding_ratio (float): Ratio of width to use as side padding (default 0.1 = 10%).

    Returns:
    - letterboxed_image (numpy.ndarray): The letterboxed image.
    """
    image_np = np.array(image)
    orig_height, orig_width = image_np.shape[:2]
    
    # print(f"Original Image Size: {orig_width}x{orig_height}")  # Before letterboxing

    # Calculate horizontal padding
    side_padding = round(orig_width * side_padding_ratio)
    padded_width = orig_width + 2 * side_padding
    padded_height = orig_height

    # print(f"Image after adding side padding: {padded_width}x{padded_height}")  # After side padding

    # Create new canvas with horizontal padding
    padded_image = np.full((padded_height, padded_width, 3), padding_color, dtype=np.uint8)
    padded_image[:, side_padding:side_padding+orig_width] = image

    # Now make it square by adding vertical padding if needed
    max_dim = max(padded_width, padded_height)
    letterboxed_image = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)

    # print(f"Image after letterboxing (square): {max_dim}x{max_dim}")  # After making square
    
    # Compute top-left corner to place padded image
    x_offset = (max_dim - padded_width) // 2
    y_offset = (max_dim - padded_height) // 2

    # Place padded image onto the final square canvas
    letterboxed_image[y_offset:y_offset+padded_height, x_offset:x_offset+padded_width] = padded_image

    return letterboxed_image

# ────────────────────────────────────────────────────────────────────────────────
# Extra helpers (put near your other metrics)
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_at_threshold(model, val_loader, device, t: float, eps: float = 1e-7):
    model.eval()
    TP = FP = FN = TN = 0
    for data, target, _ in val_loader:   # your val loader yields (img, mask, tag)
        data, target = data.to(device), target.to(device)
        logits = model(data)
        probs  = torch.sigmoid(logits)
        pred01 = (probs >= t).float()

        tp = (pred01 * target).sum((1,2,3))
        fp = (pred01 * (1 - target)).sum((1,2,3))
        fn = ((1 - pred01) * target).sum((1,2,3))
        tn = (((1 - pred01) * (1 - target))).sum((1,2,3))

        TP += tp.sum().item(); FP += fp.sum().item()
        FN += fn.sum().item(); TN += tn.sum().item()

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    iou       = TP / (TP + FP + FN + eps)
    dice      = (2*TP) / (2*TP + FP + FN + eps)
    specificity = TN / (TN + FP + eps)
    accuracy    = (TP + TN) / (TP + TN + FP + FN + eps)
    return dict(threshold=t, precision=precision, recall=recall, iou=iou,
                dice=dice, specificity=specificity, accuracy=accuracy)

@torch.no_grad()
def sweep_best_threshold(model, val_loader, device,
                         thresholds=torch.linspace(0.05, 0.95, steps=19)):
    best = {"dice": -1.0, "threshold": 0.5}
    for t in thresholds.tolist():
        m = evaluate_at_threshold(model, val_loader, device, t)
        if m["dice"] > best["dice"]:
            best = m
    return best

# ────────────────────────────────────────────────────────────────────────────────
# Device
# ────────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

DEVICE = get_device()
USE_AMP = DEVICE.type == "cuda"

# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
class SegmDataset(Dataset):
    """
    Assumes:
      - Images under IMAGES_ROOT/session_xxx/*.jpeg
      - Masks  under MASKS_ROOT/session_xxx/*.png
    Masks were created from letterboxed+resized images; here we
    reproduce the SAME letterbox on the raw image, then resize both to IMG_SIZE.
    """
    def __init__(self, images_root: Path, masks_root: Path, transform=None):
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root)
        self.transform = transform
        self.samples = self._gather_pairs()
        print(f"Found {len(self.samples)} image-mask pairs.")

    def _gather_pairs(self) -> List[Dict]:
        pairs = []
        for session_dir in sorted(self.images_root.iterdir()):
            if not session_dir.is_dir():
                continue
            mask_session = self.masks_root / session_dir.name
            for img_file in session_dir.glob("*.jpeg"):
                mask_file = mask_session / (img_file.stem + ".png")
                if mask_file.exists():
                    pairs.append({"img": img_file, "mask": mask_file, "session": session_dir.name})
                else:
                    print(f"⚠️ Missing mask for {img_file}")
        return pairs

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(str(item["img"]))[:, :, ::-1]  # BGR->RGB
        mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)

        # reproduce your mask-generation preprocessing on the image
        img = letterbox_image_with_side_padding(img, padding_color=(0,0,0),
                                                side_padding_ratio=SIDE_PADDING_RATIO)
        
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # Albumentations pipeline handles resize/normalization for BOTH image and mask
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img_t  = aug["image"]                        # [3,H,W] float tensor
            mask_t = aug["mask"].unsqueeze(0) / 255.0    # [1,H,W] in {0,1}
        else:
            img_t = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()/255.0

        return img_t, mask_t, item["session"]

class SegmDatasetTagged(Dataset):
    """
    Pairs images & masks from a single root pair, and tags them as pseudo or clean.
    Keeps the (sub)folder structure; matches files by stem.
    """
    def __init__(self, img_root: Path, mask_root: Path, is_pseudo: bool, transform=None):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.is_pseudo = is_pseudo
        self.transform = transform
        self.samples = self._gather_pairs()

        print(f"[{('PSEUDO' if is_pseudo else 'CLEAN')}] Found {len(self.samples)} pairs under "
              f"{self.img_root.name}/{self.mask_root.name}")

    def _gather_pairs(self) -> List[Dict]:
        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        imgs = sorted([p for p in self.img_root.rglob("*") if p.suffix.lower() in exts])
        pairs = []
        for ip in imgs:
            # mask lives at the same relative path (keep structure), same stem, .png
            rel = ip.relative_to(self.img_root)
            m = (self.mask_root / rel).with_suffix(".png")
            if m.exists():
                pairs.append({"img": ip, "mask": m})
        return pairs

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(str(item["img"]))[:, :, ::-1]  # BGR->RGB
        msk = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)

        # letterbox → resize (same as your training)
        img = letterbox_image_with_side_padding(img, padding_color=(0,0,0),
                                                side_padding_ratio=SIDE_PADDING_RATIO)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img_t  = aug["image"]
            mask_t = aug["mask"].unsqueeze(0) / 255.0
        else:
            img_t  = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
            mask_t = torch.from_numpy(msk).unsqueeze(0).float()/255.0

        # tag
        tag = torch.tensor(1.0 if self.is_pseudo else 0.0, dtype=torch.float32)
        return img_t, mask_t, tag

# ────────────────────────────────────────────────────────────────────────────────
# Transforms (domain-focused)
# ────────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.GaussNoise(p=0.15),
    A.MotionBlur(blur_limit=3, p=0.15),
    A.RandomScale(scale_limit=0.12, p=0.5),
    # A.RandomCrop(width=384, height=384, p=0.5),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ────────────────────────────────────────────────────────────────────────────────
# Losses & Metrics
# ────────────────────────────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum((2,3))
        denom = probs.sum((2,3)) + targets.sum((2,3))
        dice = 1 - ((2*inter + self.smooth) / (denom + self.smooth)).mean()
        return self.w * bce + (1 - self.w) * dice

def dice_metric(logits, targets, smooth=1.0, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs*targets).sum((2,3))
    denom = probs.sum((2,3)) + targets.sum((2,3))
    dice = (2*inter + smooth) / (denom + smooth + eps)
    return dice.mean().item()

def iou_metric(logits, targets, t=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > t).float()
    inter = (preds*targets).sum((2,3))
    union = preds.sum((2,3)) + targets.sum((2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def precision_recall_f1(logits, targets, t=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > t).float()
    targets = targets.float()

    tp = (preds * targets).sum((2,3))
    fp = (preds * (1 - targets)).sum((2,3))
    fn = ((1 - preds) * targets).sum((2,3))

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    # nan -> 0 (falls wirklich keine Positives im GT UND Pred)
    precision = torch.nan_to_num(precision)
    recall    = torch.nan_to_num(recall)
    f1        = torch.nan_to_num(f1)

    # Durchschnitt über Batch
    return precision.mean().item(), recall.mean().item(), f1.mean().item()

@torch.no_grad()
def find_best_threshold(model, val_loader, device, thresholds=np.linspace(0.2, 0.8, 13)):
    model.eval()
    best_t, best_d = 0.5, -1.0
    for t in thresholds:
        dices = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            # compute dice on binarized preds
            preds = (torch.sigmoid(logits) > t).float()
            inter = (preds*target).sum((2,3))
            denom = preds.sum((2,3)) + target.sum((2,3))
            d = ((2*inter + 1.0) / (denom + 1.0)).mean().item()
            dices.append(d)
        md = sum(dices)/len(dices)
        if md > best_d:
            best_d, best_t = md, t
    return best_t, best_d

class BCEDiceLossWeighted(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.w = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets, sample_weights):
        # logits, targets: [B,1,H,W]; sample_weights: [B] (clean=1.0, pseudo=PSEUDO_LOSS_WEIGHT)
        bce_map = self.bce(logits, targets)                       # [B,1,H,W]
        bce_per = bce_map.mean(dim=(1,2,3))                       # [B]

        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2,3))                  # [B,1]
        denom = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))     # [B,1]
        dice_per = 1 - ((2*inter + self.smooth) / (denom + self.smooth)).squeeze(1)  # [B]

        loss_per = self.w * bce_per + (1 - self.w) * dice_per     # [B]

        # apply sample weights, normalize by sum of weights
        sw = sample_weights.clamp_min(1e-8)                       # [B]
        loss = (loss_per * sw).sum() / sw.sum()
        return loss


# ────────────────────────────────────────────────────────────────────────────────
# Session-aware split (avoid leakage)
# ────────────────────────────────────────────────────────────────────────────────


def build_loaders(batch_size=BATCH_SIZE, val_frac=0.2, seed=42):
    # full clean set (for splitting)
    clean_full = SegmDatasetTagged(CLEAN_IMG_ROOT, CLEAN_MASK_ROOT, is_pseudo=False, transform=train_transform)

    n = len(clean_full)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_idx = idx[:n_val].tolist()
    train_clean_idx = idx[n_val:].tolist()

    # training clean subset (with train-time augs)
    train_clean = Subset(clean_full, train_clean_idx)

    # build pseudo set (train only)
    train_pseudo = SegmDatasetTagged(PSEUDO_IMG_ROOT, PSEUDO_MASK_ROOT, is_pseudo=True, transform=train_transform)

    # clamp pseudo exposure
    MAX_PSEUDO_MULT = 2.0
    if len(train_pseudo) > MAX_PSEUDO_MULT * len(train_clean_idx):
        keep = int(MAX_PSEUDO_MULT * len(train_clean_idx))
        p_idx = rng.choice(len(train_pseudo), keep, replace=False)
        train_pseudo = Subset(train_pseudo, p_idx.tolist())

    train_ds = ConcatDataset([train_clean, train_pseudo])

    # validation uses the held-out clean subset with val augs
    # rebuild a clean dataset with val transforms, then Subset it with val_idx
    clean_for_val = SegmDatasetTagged(CLEAN_IMG_ROOT, CLEAN_MASK_ROOT, is_pseudo=False, transform=val_transform)
    val_ds = Subset(clean_for_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE.type=="cuda"))
    return train_loader, val_loader



# ────────────────────────────────────────────────────────────────────────────────
# Model factory (SMP UNet with pretrained encoder)
# ────────────────────────────────────────────────────────────────────────────────
def build_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model.to(DEVICE)

# ────────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────────
def train():
    train_loader, val_loader = build_loaders(BATCH_SIZE)
    model = build_model()

    criterion = BCEDiceLossWeighted(bce_weight=0.5, smooth=1.0)

    # freeze → unfreeze exactly like before...
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW([
        {"params": model.decoder.parameters(), "lr": LR},
        {"params": model.segmentation_head.parameters(), "lr": LR}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    PHASE1_EPOCHS = 8
    best_val_dice = -1.0; epochs_no_improve = 0

    for epoch in range(EPOCHS):
        if epoch == PHASE1_EPOCHS:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.decoder.parameters(), "lr": LR},
                {"params": model.segmentation_head.parameters(), "lr": LR},
                {"params": model.encoder.parameters(), "lr": LR * 0.3}
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - PHASE1_EPOCHS)

        # ---- TRAIN ----
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        train_loss = 0.0
        for data, target, tag in pbar:   # ← tag: 0.0 clean, 1.0 pseudo
            data, target, tag = data.to(DEVICE), target.to(DEVICE), tag.to(DEVICE)
            # map tag→sample weight
            sample_w = torch.where(tag > 0.5,
                                   torch.tensor(PSEUDO_LOSS_WEIGHT, device=DEVICE),
                                   torch.tensor(1.0, device=DEVICE))

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(data)
                loss = criterion(logits, target, sample_w)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ---- VALIDATION on CLEAN ONLY ----
        model.eval()
        val_loss = 0.0
        dice_vals, iou_vals = [], []
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                logits = model(data)
                # validation uses clean weight=1 for all
                sample_w = torch.ones(data.size(0), device=DEVICE)
                loss = criterion(logits, target, sample_w)
                val_loss += loss.item()
                dice_vals.append(dice_metric(logits, target))
                iou_vals.append(iou_metric(logits, target, t=0.5))

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val   = val_loss   / max(len(val_loader), 1)
        avg_dice  = sum(dice_vals)/max(len(dice_vals), 1)
        avg_iou   = sum(iou_vals)/max(len(iou_vals), 1)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train {avg_train:.4f} | Val {avg_val:.4f} | "
              f"Dice {avg_dice:.4f} | IoU {avg_iou:.4f}")

        # early stopping logic unchanged...
        # save best on val_dice as before...

        # ---- save best on clean validation Dice & early stop ----
        if epoch == 0:
            best_val_dice = -1.0
            epochs_no_improve = 0

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_dice,
                "val_iou": avg_iou,
            }, CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no Val Dice improvement).")
                break

        scheduler.step()

    # ===================== after training loop =====================
    print("Lade bestes Checkpoint und suche besten Threshold auf CLEAN-Validation ...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # feiner Sweep
    thr_grid = torch.linspace(0.05, 0.95, steps=19)
    best = sweep_best_threshold(model, val_loader, DEVICE, thresholds=thr_grid)

    print(
        f"Best Threshold (by Dice) t={best['threshold']:.2f} | "
        f"Dice={best['dice']:.4f} IoU={best['iou']:.4f} "
        f"P={best['precision']:.4f} R={best['recall']:.4f} "
        f"Spec={best['specificity']:.4f} Acc={best['accuracy']:.4f}"
    )

    # Ergebnisse neben dem .pth sichern
    import json
    thr_json = CHECKPOINT_PATH.with_suffix(".threshold_12.json")
    thr_txt  = CHECKPOINT_PATH.with_suffix(".threshold_12.txt")
    with open(thr_json, "w") as f:
        json.dump(best, f, indent=2)
    with open(thr_txt, "w") as f:
        f.write(
            f"Best threshold (val, by Dice): {best['threshold']:.2f}\n"
            f"Dice: {best['dice']:.6f}\nIoU: {best['iou']:.6f}\n"
            f"Precision: {best['precision']:.6f}\nRecall: {best['recall']:.6f}\n"
            f"Specificity: {best['specificity']:.6f}\nAccuracy: {best['accuracy']:.6f}\n"
        )

    return model, best["threshold"]



# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
