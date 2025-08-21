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
import segmentation_models_pytorch as smp  # pip install segmentation-models-pytorch timm

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG (paths are what you showed in your last message)
# ────────────────────────────────────────────────────────────────────────────────
IMAGES_ROOT = Path("~/sarah/background_segmentation/v2_expansion").expanduser()
MASKS_ROOT  = Path("~/sarah/background_segmentation/final_masks_cleaned").expanduser()

# where to save the model + metrics
CHECKPOINT_PATH = Path("~/sarah/background_segmentation/checkpoints_pretrained/model_8.pth").expanduser()
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


# training size (try 512; lower to 384 if VRAM is tight)
IMG_SIZE: Tuple[int, int] = (512, 512)
SIDE_PADDING_RATIO = 0.1  # must match what you used when generating masks

BATCH_SIZE = 6          # adjust per VRAM (try 4–8 at 512)
EPOCHS = 50
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

# ────────────────────────────────────────────────────────────────────────────────
# Session-aware split (avoid leakage)
# ────────────────────────────────────────────────────────────────────────────────
def build_loaders(batch_size=BATCH_SIZE):
    full_ds = SegmDataset(IMAGES_ROOT, MASKS_ROOT, transform=None)  # build pairs & sessions
    # collect unique sessions
    sessions = sorted({s for _, _, s in full_ds})
    # simple 80/20 split by session
    n_val = max(1, int(0.2 * len(sessions)))
    val_sessions = set(sessions[:n_val])  # deterministic; could randomize if needed
    train_idx, val_idx = [], []
    for i in range(len(full_ds)):
        _, _, s = full_ds[i]
        (val_idx if s in val_sessions else train_idx).append(i)

    # now wrap with transforms
    class _Subset(Dataset):
        def __init__(self, base, indices, transform):
            self.base, self.indices, self.transform = base, indices, transform
        def __len__(self): return len(self.indices)

        def __getitem__(self, k):
            i = self.indices[k]
            # load raw
            raw_img = cv2.imread(str(self.base.samples[i]["img"]))[:, :, ::-1]
            raw_mask = cv2.imread(str(self.base.samples[i]["mask"]), cv2.IMREAD_GRAYSCALE)

            # letterbox image (to match how masks were created), NOT the mask
            raw_img = letterbox_image_with_side_padding(
                raw_img, padding_color=(0,0,0), side_padding_ratio=SIDE_PADDING_RATIO
            )

            # now bring BOTH to the SAME size before Albumentations
            img_resized  = cv2.resize(raw_img,  IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(raw_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            if self.transform:
                aug   = self.transform(image=img_resized, mask=mask_resized)
                img_t = aug["image"]
                mask_t = aug["mask"].unsqueeze(0) / 255.0
            else:
                img_t  = torch.from_numpy(img_resized.transpose(2,0,1)).float() / 255.0
                mask_t = torch.from_numpy(mask_resized).unsqueeze(0).float() / 255.0

            return img_t, mask_t


    train_ds = _Subset(full_ds, train_idx, train_transform)
    val_ds   = _Subset(full_ds, val_idx,   val_transform)

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
    criterion = BCEDiceLoss(bce_weight=0.5, smooth=1.0)

    # === PHASE 1: Encoder einfrieren ===
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Optimizer nur für Decoder + Head
    optimizer = optim.AdamW([
        {"params": model.decoder.parameters(), "lr": LR},
        {"params": model.segmentation_head.parameters(), "lr": LR}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_dice = -1.0
    epochs_no_improve = 0
    metrics_csv = CHECKPOINT_PATH.parent / "training_metrics_8.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch","phase","train_loss","val_loss",
                "val_dice","val_iou","val_precision","val_recall","val_f1"
            ])


    PHASE1_EPOCHS = 8  # Warmup-Dauer

    for epoch in range(EPOCHS):
        # === PHASENWECHSEL ===
        if epoch == PHASE1_EPOCHS:
            print("🔓 Unfreeze Encoder + kleinere LR für Encoder")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.decoder.parameters(), "lr": LR},
                {"params": model.segmentation_head.parameters(), "lr": LR},
                {"params": model.encoder.parameters(), "lr": LR * 0.3}  # Encoder kleineres LR
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - PHASE1_EPOCHS)

        phase = 1 if epoch < PHASE1_EPOCHS else 2

        # === TRAIN ===
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train] (Phase {phase})")
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(data)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        dice_vals, iou_vals = [], []
        prec_vals, rec_vals, f1_vals = [], [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                logits = model(data)
                loss = criterion(logits, target)
                val_loss += loss.item()

                dice_vals.append(dice_metric(logits, target))
                iou_vals.append(iou_metric(logits, target, t=0.5))

                p, r, f1 = precision_recall_f1(logits, target, t=0.5)
                prec_vals.append(p)
                rec_vals.append(r)
                f1_vals.append(f1)

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val   = val_loss   / max(len(val_loader), 1)
        avg_dice  = sum(dice_vals)/max(len(dice_vals), 1)
        avg_iou   = sum(iou_vals)/max(len(iou_vals), 1)
        avg_prec = sum(prec_vals) / max(len(prec_vals), 1)
        avg_rec  = sum(rec_vals)  / max(len(rec_vals), 1)
        avg_f1   = sum(f1_vals)   / max(len(f1_vals), 1)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Phase {phase} | "
            f"Train {avg_train:.4f} | Val {avg_val:.4f} | "
            f"Dice {avg_dice:.4f} | IoU {avg_iou:.4f} | "
            f"P {avg_prec:.4f} | R {avg_rec:.4f} | F1 {avg_f1:.4f}"
        )

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, phase, avg_train, avg_val,
                avg_dice, avg_iou, avg_prec, avg_rec, avg_f1
            ])


        # Best Model speichern
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
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


    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    best_t, best_d = find_best_threshold(model, val_loader, DEVICE)
    print(f"Best threshold: {best_t:.2f} | Val Dice (bin): {best_d:.4f}")
    return model, best_t, val_loader

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
