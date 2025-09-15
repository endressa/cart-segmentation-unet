#!/usr/bin/env python3
"""
Fine-tune a segmentation model (Unet + EfficientNet-B0 encoder) on two datasets:
- CLEAN: images + corresponding masks
- HARD:  images + corresponding masks

It loads INIT_CKPT if present and writes the best fine-tuned weights to OUT_CKPT.
Basic early stopping on validation Dice. Metrics are appended to METRICS_FILE (JSON).

Dependencies (pip):
  - torch, torchvision, tqdm, albumentations, segmentation-models-pytorch, timm, opencv-python, numpy

Usage:
  python finetune_clean_and_hard.py
"""
from pathlib import Path
import os
import json
import random
from typing import List, Tuple, Dict

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
# External loss dependency (from your repo): losses/abl.py
# ────────────────────────────────────────────────────────────────────────────────
try:
    from losses.abl import ABL
except Exception as e:
    raise SystemExit("Could not import ABL from losses.abl — make sure your repo is on PYTHONPATH.")

# ────────────────────────────────────────────────────────────────────────────────
# PATHS (provided)
# ────────────────────────────────────────────────────────────────────────────────
CLEAN_IMAGES = Path("/home/ansible/sarah/background_segmentation/dataset/images_cleaned")
CLEAN_MASKS  = Path("/home/ansible/sarah/background_segmentation/dataset/final_masks_cleaned")
HARD_IMAGES  = Path("/home/ansible/sarah/background_segmentation/dataset/images_hard")
HARD_MASKS   = Path("/home/ansible/sarah/background_segmentation/dataset/masks_hard")

INIT_CKPT = Path("/home/ansible/sarah/background_segmentation/checkpoints_pretrained/BCEDiceABL_pseudo.pth")
OUT_CKPT  = Path("/home/ansible/sarah/background_segmentation/checkpoints_pretrained/finetuned_BCEDiceABL_pseudo_early_stop.pth")
METRICS_FILE = OUT_CKPT.with_suffix(".metrics.json")
OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────────
IMG_SIZE: Tuple[int, int] = (512, 512)
SIDE_PADDING_RATIO = 0.10  # must match your preprocessing during training/inference
BATCH_SIZE = 6
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8  # early stopping on val dice
VAL_FRACTION = 0.2  # session-aware split
RANDOM_SEED = 1337

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ────────────────────────────────────────────────────────────────────────────────
# Repro
# ────────────────────────────────────────────────────────────────────────────────
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ────────────────────────────────────────────────────────────────────────────────
# Device & AMP
# ────────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

DEVICE = get_device()
USE_AMP = DEVICE.type == "cuda"

# ────────────────────────────────────────────────────────────────────────────────
# Preprocessing helper: letterbox + side padding (to match predictor)
# ────────────────────────────────────────────────────────────────────────────────
def letterbox_image_with_side_padding(image, padding_color=(0, 0, 0), side_padding_ratio=SIDE_PADDING_RATIO):
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    side_padding = round(w * side_padding_ratio)
    padded_w = w + 2 * side_padding
    padded_h = h

    padded = np.full((padded_h, padded_w, 3), padding_color, dtype=np.uint8)
    padded[:, side_padding:side_padding+w] = image_np

    max_dim = max(padded_w, padded_h)
    canvas = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)
    x_off = (max_dim - padded_w) // 2
    y_off = (max_dim - padded_h) // 2
    canvas[y_off:y_off+padded_h, x_off:x_off+padded_w] = padded
    return canvas

# ────────────────────────────────────────────────────────────────────────────────
# Session key extraction (to avoid leakage between train/val)
# ────────────────────────────────────────────────────────────────────────────────
def derive_session_key(rel_path: Path) -> str:
    parts = list(rel_path.parts)
    for i, comp in enumerate(parts):
        if comp.startswith("session_"):
            prev = parts[i-1] if i-1 >= 0 else ""
            if prev.startswith("store_"):
                return f"{prev}/{comp}"
            return comp
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if len(parts) == 1:
        return parts[0]
    return "unknown_session"

# ────────────────────────────────────────────────────────────────────────────────
# Dataset that merges multiple (images_root, masks_root) pairs
# ────────────────────────────────────────────────────────────────────────────────
class PairRootSegDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, pairs: List[Tuple[Path, Path]]):
        self.pairs = [(Path(a), Path(b)) for a, b in pairs]
        self.samples: List[Dict] = self._gather()
        print(f"Found {len(self.samples)} image–mask pairs across {len(self.pairs)} roots.")

    def _gather(self) -> List[Dict]:
        out: List[Dict] = []
        missing = 0
        for img_root, msk_root in self.pairs:
            if not img_root.exists():
                print(f"⚠️ Images root does not exist: {img_root}")
                continue
            for img in img_root.rglob("*"):
                if img.suffix.lower() not in self.IMG_EXTS:
                    continue
                rel = img.relative_to(img_root)
                mask = (msk_root / rel).with_suffix(".png")
                if mask.exists():
                    sess = derive_session_key(rel)
                    out.append({"img": img, "mask": mask, "session": sess})
                else:
                    missing += 1
        if missing:
            print(f"⚠️ Missing masks for {missing} images (skipped).")
        random.shuffle(out)
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img = cv2.imread(str(rec["img"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {rec['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(rec["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {rec['mask']}")

        img = letterbox_image_with_side_padding(img)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        return img, mask, rec["session"]

# ────────────────────────────────────────────────────────────────────────────────
# Transforms
# ────────────────────────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.GaussNoise(p=0.15),
    A.MotionBlur(blur_limit=3, p=0.15),
    A.RandomScale(scale_limit=0.12, p=0.5),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ────────────────────────────────────────────────────────────────────────────────
# Loss & metrics
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
        inter = (probs*targets).sum((2,3))
        denom = probs.sum((2,3)) + targets.sum((2,3))
        dice = 1 - ((2*inter + self.smooth) / (denom + self.smooth)).mean()
        return self.w*bce + (1-self.w)*dice

@torch.no_grad()
def dice_metric(logits, targets, smooth=1.0, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs*targets).sum((2,3))
    denom = probs.sum((2,3)) + targets.sum((2,3))
    dice = (2*inter + smooth) / (denom + smooth + eps)
    return dice.mean().item()

class BCEDiceABL(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0, abl_weight=0.1):
        super().__init__()
        self.region_loss = BCEDiceLoss(bce_weight=bce_weight, smooth=smooth)
        self.boundary_loss = ABL()
        self.abl_weight = abl_weight
        self.boundary_none_count = 0
        self.total_calls = 0
    def components(self, logits, targets):
        region = self.region_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        self.total_calls += 1
        if boundary is None:
            self.boundary_none_count += 1
            total = region
            boundary_val = torch.tensor(0.0, device=logits.device)
        else:
            total = region + self.abl_weight * boundary
            boundary_val = boundary.detach()
        if self.total_calls % 500 == 0:
            ratio = self.boundary_none_count / self.total_calls
            print(f"[ABL Debug] boundary=None {self.boundary_none_count}/{self.total_calls} ({ratio:.1%})")
        return {"total": total, "region": region.detach(), "boundary": boundary_val}
    def forward(self, logits, targets):
        return self.components(logits, targets)["total"]

# ────────────────────────────────────────────────────────────────────────────────
# Dataloaders with session-aware split
# ────────────────────────────────────────────────────────────────────────────────

def build_loaders(batch_size=BATCH_SIZE):
    full = PairRootSegDataset([
        (CLEAN_IMAGES, CLEAN_MASKS),
        (HARD_IMAGES,  HARD_MASKS),
    ])
    sessions = sorted({rec["session"] for rec in full.samples})
    if not sessions:
        raise SystemExit("No sessions found — check your paths.")
    random.shuffle(sessions)
    n_val = max(1, int(len(sessions) * VAL_FRACTION))
    val_sessions = set(sessions[:n_val])
    train_sessions = set(sessions[n_val:])

    train_idx, val_idx = [], []
    for i, rec in enumerate(full.samples):
        (val_idx if rec["session"] in val_sessions else train_idx).append(i)

    print(f"Sessions total: {len(sessions)} | train: {len(train_sessions)} | val: {len(val_sessions)}")
    print(f"Samples: train {len(train_idx)} | val {len(val_idx)}")

    class _Subset(Dataset):
        def __init__(self, base: PairRootSegDataset, indices: List[int], transform):
            self.base, self.indices, self.transform = base, indices, transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, k):
            img, mask, _ = self.base[self.indices[k]]
            if self.transform:
                aug = self.transform(image=img, mask=mask)
                img_t = aug["image"]
                mask_t = aug["mask"].unsqueeze(0) / 255.0
            else:
                img_t = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
                mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
            return img_t, mask_t

    train_ds = _Subset(full, train_idx, train_transform)
    val_ds   = _Subset(full, val_idx,   val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=(DEVICE.type=="cuda"))
    return train_loader, val_loader

# ────────────────────────────────────────────────────────────────────────────────
# Model
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
# Checkpoint helpers
# ────────────────────────────────────────────────────────────────────────────────

def load_init_weights(model: nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        print(f"⚠️ INIT_CKPT does not exist at {ckpt_path}; training from imagenet encoder.")
        return
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded INIT_CKPT. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        if missing:
            print("  Missing keys (first 10):", list(missing)[:10])
        if unexpected:
            print("  Unexpected keys (first 10):", list(unexpected)[:10])
    except Exception as e:
        print(f"⚠️ Failed to load INIT_CKPT: {e}")


def save_best(model: nn.Module, epoch: int, val_dice: float):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_dice": val_dice,
    }, OUT_CKPT)
    print(f"  ✔ Saved new best model to {OUT_CKPT} (Dice {val_dice:.4f})")

# ────────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────────

def train():
    train_loader, val_loader = build_loaders(BATCH_SIZE)
    model = build_model()
    load_init_weights(model, INIT_CKPT)

    criterion = BCEDiceABL(bce_weight=0.5, smooth=1.0, abl_weight=0.1)

    # Phase 1: freeze encoder (warmup)
    for p in model.encoder.parameters():
        p.requires_grad = False

    opt = optim.AdamW([
        {"params": model.decoder.parameters(),           "lr": LR},
        {"params": model.segmentation_head.parameters(), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {"epochs": []}
    if METRICS_FILE.exists():
        try:
            history = json.loads(METRICS_FILE.read_text())
        except Exception:
            history = {"epochs": []}

    PHASE1_EPOCHS = 8

    for epoch in range(1, EPOCHS + 1):
        phase = 1 if epoch <= PHASE1_EPOCHS else 2
        if epoch == PHASE1_EPOCHS + 1:
            # unfreeze encoder at smaller LR
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = optim.AdamW([
                {"params": model.encoder.parameters(),           "lr": LR * 0.3},
                {"params": model.decoder.parameters(),           "lr": LR},
                {"params": model.segmentation_head.parameters(), "lr": LR},
            ], weight_decay=WEIGHT_DECAY)

        # ---- TRAIN ----
        model.train()
        tr_tot = tr_reg = tr_bnd = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train] (Phase {phase})")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                comps = criterion.components(logits=model(imgs), targets=masks)
                loss = comps["total"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_tot += float(loss.item())
            tr_reg += float(comps["region"].item())
            tr_bnd += float(comps["boundary"].item())
            pbar.set_postfix({
                "total": f"{comps['total'].item():.4f}",
                "region": f"{comps['region'].item():.4f}",
                "boundary": f"{comps['boundary'].item():.4f}",
            })

        # ---- VAL ----
        model.eval()
        va_tot = va_reg = va_bnd = 0.0
        val_dice_vals = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                comps = criterion.components(logits=logits, targets=masks)
                va_tot += float(comps["total"].item())
                va_reg += float(comps["region"].item())
                va_bnd += float(comps["boundary"].item())
                val_dice_vals.append(dice_metric(logits, masks))

        ntr, nva = max(1, len(train_loader)), max(1, len(val_loader))
        avg_train_total = tr_tot / ntr
        avg_train_region = tr_reg / ntr
        avg_train_boundary = tr_bnd / ntr
        avg_val_total = va_tot / nva
        avg_val_region = va_reg / nva
        avg_val_boundary = va_bnd / nva
        avg_dice = sum(val_dice_vals) / max(1, len(val_dice_vals))
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{EPOCHS} | Phase {phase} | LR {lr_now:.2e} | "
            f"Train T/R/B {avg_train_total:.4f}/{avg_train_region:.4f}/{avg_train_boundary:.4f} | "
            f"Val T/R/B {avg_val_total:.4f}/{avg_val_region:.4f}/{avg_val_boundary:.4f} | "
            f"Dice {avg_dice:.4f}"
        )

        # JSON history (append + overwrite)
        epoch_rec = {
            "epoch": epoch, "phase": phase, "lr": lr_now,
            "train": {"total": avg_train_total, "region": avg_train_region, "boundary": avg_train_boundary},
            "val":   {"total": avg_val_total,   "region": avg_val_region,   "boundary": avg_val_boundary},
            "val_dice": avg_dice,
        }
        history["epochs"].append(epoch_rec)
        METRICS_FILE.write_text(json.dumps(history, indent=2))

        # checkpoint on validation loss (lower is better)
        current = avg_val_total  # or avg_val_region if you prefer just region; your call
        if current < (best_val_loss):
            best_val_loss = current
            epochs_no_improve = 0
            # Save and also record the loss in the checkpoint for traceability
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_total_loss": best_val_loss,
                "val_dice": avg_dice,  # keep for info
            }, OUT_CKPT)
            print(f"  ✔ Saved new best model to {OUT_CKPT} (Val loss {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no Val loss improvement).")
                break


    print(f"Best Val loss: {best_val_loss:.6f}")
    return OUT_CKPT

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
