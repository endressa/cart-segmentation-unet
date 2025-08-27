# finetune_hard_only.py
import os, json, math, random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------------- PATHS (your values) ----------------
CHECKPOINT_PATH = Path("~/sarah/background_segmentation/checkpoints_pretrained/pseudo_model_12.pth").expanduser()
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

# hard-corrected data (letterboxed 512×512 PNG masks)
HARD_IMG_ROOT  = Path("~/sarah/background_segmentation/dataset/images_hard").expanduser()
HARD_MASK_ROOT = Path("~/sarah/background_segmentation/dataset/masks_hard").expanduser()

# output checkpoint
CKPT_IN  = CHECKPOINT_PATH                                  
CKPT_OUT = CHECKPOINT_PATH.with_name("finetuned_model_12.pth")

# ---------------- TRAINING KNOBS ----------------
IMG_SIZE = (512, 512)
SIDE_PADDING_RATIO = 0.1      # keep consistent with training
USE_IMAGENET_NORM = True      # True = ImageNet mean/std, matches your main training
BATCH_SIZE = 8
EPOCHS = 12

# --- NEW: split LRs + warmup ---
HEAD_LR = 3e-5       # decoder + segmentation head
ENCODER_LR = 1e-5    # encoder (≈ 0.3 × head LR)
WARMUP_EPOCHS = 3    # freeze encoder for first N epochs

WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42
NUM_WORKERS = 2
USE_AMP = True

# ---------------- PREPROCESS ----------------
def letterbox_image_with_side_padding(image, padding_color=(0,0,0), side_padding_ratio=SIDE_PADDING_RATIO):
    """Replicate training-time letterbox (horizontal side padding -> square canvas)."""
    h, w = image.shape[:2]
    side_padding = round(w * side_padding_ratio)
    padded_w = w + 2 * side_padding
    padded_h = h
    padded = np.full((padded_h, padded_w, 3), padding_color, dtype=np.uint8)
    padded[:, side_padding:side_padding+w] = image
    max_dim = max(padded_w, padded_h)
    canvas = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)
    xo = (max_dim - padded_w)//2
    yo = (max_dim - padded_h)//2
    canvas[yo:yo+padded_h, xo:xo+padded_w] = padded
    return canvas

def make_transform(train=True):
    if USE_IMAGENET_NORM:
        mean=(0.485, 0.456, 0.406); std=(0.229, 0.224, 0.225)
    else:
        mean=(0,0,0); std=(1,1,1)
    aug = [A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_LINEAR)]
    # (optional) add light jitter/flip here if you want
    aug += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(aug)


# ---- Helpers for differential LRs and encoder warmup ----
def build_param_groups_for_smp(model, encoder_lr, head_lr, weight_decay=0.0):
    # Split params: encoder vs (decoder + segmentation_head)
    enc_params = []
    if hasattr(model, "encoder"):
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = []
    if hasattr(model, "decoder"):
        head_params += list(model.decoder.parameters())
    if hasattr(model, "segmentation_head"):
        head_params += list(model.segmentation_head.parameters())
    # Deduplicate (safety)
    seen = set(); uniq_head = []
    for p in head_params:
        if id(p) not in seen:
            seen.add(id(p)); uniq_head.append(p)
    return [
        {"params": enc_params, "lr": float(encoder_lr), "weight_decay": weight_decay},
        {"params": uniq_head, "lr": float(head_lr), "weight_decay": weight_decay},
    ]

class EncoderWarmupController:
    def __init__(self, model, warmup_epochs:int):
        self.model = model
        self.warmup_epochs = max(int(warmup_epochs), 0)
        # Freeze encoder initially
        if self.warmup_epochs > 0 and hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False

    def maybe_unfreeze(self, current_epoch:int, optimizer):
        # Unfreeze exactly at the start of epoch == warmup_epochs
        if self.warmup_epochs == 0 or not hasattr(self.model, "encoder"):
            return
        if current_epoch == self.warmup_epochs:
            for p in self.model.encoder.parameters():
                p.requires_grad = True
            # Ensure optimizer tracks newly trainable params
            for g in optimizer.param_groups:
                g["params"] = [p for p in g["params"] if p.requires_grad]

# ---------------- DATASET ----------------
class HardSegmDataset(Dataset):
    """
    Expects images under HARD_IMG_ROOT and masks under HARD_MASK_ROOT,
    same relative structure; masks are 512×512 letterboxed PNGs (0/255).
    """
    def __init__(self, img_root: Path, mask_root: Path, transform=None):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.transform = transform
        image_exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        imgs = [p for p in self.img_root.rglob("*") if p.suffix.lower() in image_exts]
        pairs = []
        for ip in imgs:
            rel = ip.relative_to(self.img_root)
            mp = (self.mask_root / rel).with_suffix(".png")
            if mp.exists():
                pairs.append((ip, mp))
        if not pairs:
            raise RuntimeError(f"No image/mask pairs found under {img_root} / {mask_root}")
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read image: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb  = letterbox_image_with_side_padding(img_rgb)

        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mp}")
        # ensure 0/255, resize to 512 (NEAREST) just in case
        mask = (mask > 127).astype(np.uint8) * 255
        if mask.shape[:2] != IMG_SIZE:
            mask = cv2.resize(mask, IMG_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

        augmented = self.transform(image=img_lb)
        x = augmented["image"]             # C×H×W
        y = torch.from_numpy((mask/255.0).astype(np.float32)).unsqueeze(0)  # 1×H×W
        return x, y, torch.tensor(1.0, dtype=torch.float32)  # weight 1.0 (not used, keeps signature)

# ---------------- MODEL/LOSS/METRICS ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)
    return model

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1,2,3))
        den = (probs + targets).sum(dim=(1,2,3))
        dice = 1 - (2*inter + self.smooth) / (den + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()

@torch.no_grad()
def dice_metric(logits, targets, t=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > t).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    den = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    d = (2*inter + eps) / (den + eps)
    return float(d.mean().item())

@torch.no_grad()
def iou_metric(logits, targets, t=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > t).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - inter
    i = (inter + eps) / (union + eps)
    return float(i.mean().item())

@torch.no_grad()
def sweep_best_threshold(model, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, steps=19)
    model.eval()
    metrics = []
    for t in thresholds:
        dices, ious = [], []
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            dices.append(dice_metric(logits, y, t=float(t)))
            ious.append(iou_metric(logits, y, t=float(t)))
        metrics.append((float(t), float(np.mean(dices)), float(np.mean(ious))))
    # pick by best Dice
    best = max(metrics, key=lambda z: z[1])
    return {"threshold": best[0], "dice": best[1], "iou": best[2]}

# ---------------- DATALOADERS (hard-only split) ----------------
def build_hard_loaders(val_frac=0.1):
    ds = HardSegmDataset(HARD_IMG_ROOT, HARD_MASK_ROOT, transform=make_transform(train=True))
    n = len(ds)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    return train_loader, val_loader

# ---------------- FINETUNE ----------------
def finetune_on_hard():
    train_loader, val_loader = build_hard_loaders(val_frac=0.1)

    model = build_model()
    ckpt = torch.load(CKPT_IN, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])   # start from model 13
    model.train()

    # --- NEW: set up encoder warmup (freezes now if WARMUP_EPOCHS > 0)
    encoder_warmup = EncoderWarmupController(model, WARMUP_EPOCHS)

    optimizer = optim.AdamW(
        build_param_groups_for_smp(model, ENCODER_LR, HEAD_LR, weight_decay=WEIGHT_DECAY),
        lr=HEAD_LR,  # base LR; per-group LRs above take precedence
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    criterion = BCEDiceLoss(bce_weight=0.5, smooth=1.0)

    best_val_dice = -1.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        encoder_warmup.maybe_unfreeze(epoch, optimizer)
        # ---- TRAIN ----
        model.train()
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{EPOCHS} [Train]")
        total = 0.0
        for x, y, _ in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ---- VALIDATE ----
        model.eval()
        vloss, dices, ious = 0.0, [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                vloss += criterion(logits, y).item()
                dices.append(dice_metric(logits, y, t=0.5))
                ious.append(iou_metric(logits, y, t=0.5))

        avg_train = total / max(1, len(train_loader))
        avg_val   = vloss / max(1, len(val_loader))
        avg_dice  = float(np.mean(dices))
        avg_iou   = float(np.mean(ious))
        print(f"[FT] Epoch {epoch+1}/{EPOCHS} | Train {avg_train:.4f} | Val {avg_val:.4f} | Dice {avg_dice:.4f} | IoU {avg_iou:.4f}")

        # early stopping on best Val Dice
        improved = avg_dice > best_val_dice
        if improved:
            best_val_dice = avg_dice
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_dice,
                "val_iou": avg_iou,
            }, CKPT_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[FT] Early stopping (no Val Dice improvement {PATIENCE} epochs).")
                break

        scheduler.step()

    # ---- Load best & threshold sweep ----
    best = torch.load(CKPT_OUT, map_location=DEVICE)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    metrics = sweep_best_threshold(model, val_loader, DEVICE)
    print(f"[FT] Best t={metrics['threshold']:.2f} | Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f}")

    thr_json = CKPT_OUT.with_suffix(".threshold.json")
    with open(thr_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[FT] Saved: {CKPT_OUT} and {thr_json}")

    return model, metrics["threshold"]

# ---------------- MAIN ----------------
if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    finetune_on_hard()
