#!/usr/bin/env python3
"""
train_pseudo.py
Train on all cleaned pseudo-labels with a session-aware split.

- Images come from multiple roots (same relative sub-tree as masks)
- Masks live in a single merged root with identical subfolder structure
- Session-aware split: ensures train/val don't share near-duplicate frames from the same session
"""
from losses.abl import ABL
import json
import os
import random
from pathlib import Path
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
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────
IMAGE_ROOTS = [
    Path("/opt/whizcart/shared/carrefour_classes/images/merci"),
    Path("/opt/whizcart/shared/carrefour_classes/images/gemuese_netz"),
    Path("/opt/whizcart/shared/carrefour_classes/images/head_and_shoulders_sub_sarah"),
]
MASKS_ROOT = Path("/home/ansible/sarah/background_segmentation/dataset/mixed_pseudo_clean")

CHECKPOINT_PATH = Path("~/sarah/background_segmentation/checkpoints_pretrained/BCEDiceABL_pseudo.pth").expanduser()
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_JSON = CHECKPOINT_PATH.with_suffix(".metrics.json")

IMG_SIZE: Tuple[int, int] = (512, 512)
SIDE_PADDING_RATIO = 0.1  # must match generation/cleaning
BATCH_SIZE = 6
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8  # early stopping on val dice
VAL_FRACTION = 0.2
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
# Preprocessing: letterbox with side padding (matches your predictor)
# ────────────────────────────────────────────────────────────────────────────────
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
# Session key extraction
# ────────────────────────────────────────────────────────────────────────────────
def derive_session_key(rel_path: Path) -> str:
    """
    Try to produce a stable session key from a relative path.

    Heuristics:
    - If path contains a component starting with 'session_', pair it with the preceding 'store_*' if present:
      e.g. raw/store_<uuid>/session_<uuid>/...  ->  store_<uuid>/session_<uuid>
    - Else, use the first two components if available
    - Else, use the parent folder name
    """
    parts = list(rel_path.parts)
    # try store/session pattern
    for i, comp in enumerate(parts):
        if comp.startswith("session_"):
            prev = parts[i-1] if i-1 >= 0 else ""
            if prev.startswith("store_"):
                return f"{prev}/{comp}"
            return comp
    # fallback: first two levels
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if len(parts) == 1:
        return parts[0]
    return "unknown_session"

# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
class PseudoSegmDataset(Dataset):
    """
    - Looks for images under multiple roots.
    - Looks for masks under a single MASKS_ROOT with identical relative structure.
    - Records a 'session_key' per sample for leakage-safe splitting.
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, image_roots: List[Path], masks_root: Path):
        self.image_roots = [Path(r) for r in image_roots]
        self.masks_root = Path(masks_root)
        self.samples: List[Dict] = self._gather_pairs()
        print(f"Found {len(self.samples)} image–mask pairs across {len(self.image_roots)} roots.")

    def _gather_pairs(self) -> List[Dict]:
        pairs: List[Dict] = []
        missing = 0
        for root in self.image_roots:
            if not root.exists():
                print(f"⚠️ Images root does not exist: {root}")
                continue
            for img in root.rglob("*"):
                if img.suffix.lower() not in self.IMG_EXTS:
                    continue
                rel = img.relative_to(root)
                mask = (self.masks_root / rel).with_suffix(".png")
                if mask.exists():
                    session_key = derive_session_key(rel)
                    pairs.append({"img": img, "mask": mask, "session": session_key})
                else:
                    missing += 1
        if missing:
            print(f"⚠️ Missing masks for {missing} images (skipped).")
        # Optional: shuffle pairs for variety (deterministic via seed)
        random.shuffle(pairs)
        return pairs

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img = cv2.imread(str(rec["img"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {rec['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(rec["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {rec['mask']}")

        # reproduce predictor preprocessing on image (not on mask)
        img = letterbox_image_with_side_padding(img)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # return raw arrays; transforms applied in Subset wrapper
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
        self.boundary_loss = ABL(num_classes=1)  # binary segmentation
        self.abl_weight = abl_weight


    def components(self, logits, targets):
        """Return dict with separate loss parts + total (no detach on total)."""
        region = self.region_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        total = region + self.abl_weight * boundary
        return {"total": total, "region": region.detach(), "boundary": boundary.detach()}

    def forward(self, logits, targets):
        return self.components(logits, targets)["total"]

# ────────────────────────────────────────────────────────────────────────────────
# Build loaders with session-aware split
# ────────────────────────────────────────────────────────────────────────────────
def build_loaders(batch_size=BATCH_SIZE):
    full = PseudoSegmDataset(IMAGE_ROOTS, MASKS_ROOT)
    # gather sessions
    sessions = sorted({rec["session"] for rec in full.samples})
    if not sessions:
        raise SystemExit("No sessions found — check roots/paths.")
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
        def __init__(self, base: PseudoSegmDataset, indices: List[int], transform):
            self.base, self.indices, self.transform = base, indices, transform
        def __len__(self): return len(self.indices)
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
# Train
# ────────────────────────────────────────────────────────────────────────────────
def train():
    train_loader, val_loader = build_loaders(BATCH_SIZE)
    model = build_model()
    criterion = BCEDiceABL(bce_weight=0.5, smooth=1.0, abl_weight=0.1)

    # Phase 1: freeze encoder (warmup)
    for p in model.encoder.parameters():
        p.requires_grad = False

    opt = optim.AdamW([
        {"params": model.decoder.parameters(),          "lr": LR},
        {"params": model.segmentation_head.parameters(),"lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_val_dice = -1.0
    epochs_no_improve = 0

    history = {"epochs": []}
    if METRICS_JSON.exists():
        try:
            history = json.loads(METRICS_JSON.read_text())
        except Exception:
            history = {"epochs": []}


    PHASE1_EPOCHS = 8

    for epoch in range(EPOCHS):
        phase = 1 if epoch < PHASE1_EPOCHS else 2
        if epoch == PHASE1_EPOCHS:
            # unfreeze encoder with smaller LR
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = optim.AdamW([
                {"params": model.encoder.parameters(),          "lr": LR * 0.3},
                {"params": model.decoder.parameters(),          "lr": LR},
                {"params": model.segmentation_head.parameters(),"lr": LR},
            ], weight_decay=WEIGHT_DECAY)


        # ---- TRAIN ----
        model.train()
        tr_tot = tr_reg = tr_bnd = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train] (Phase {phase})")
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
        # current lr (take first group)
        lr_now = opt.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{EPOCHS} | Phase {phase} | "
              f"LR {lr_now:.2e} | "
              f"Train T/R/B {avg_train_total:.4f}/{avg_train_region:.4f}/{avg_train_boundary:.4f} | "
              f"Val T/R/B {avg_val_total:.4f}/{avg_val_region:.4f}/{avg_val_boundary:.4f} | "
              f"Dice {avg_dice:.4f}")


        # JSON history (append + overwrite)
        epoch_rec = {
            "epoch": epoch+1, "phase": phase, "lr": lr_now,
            "train": {"total": avg_train_total, "region": avg_train_region, "boundary": avg_train_boundary},
            "val":   {"total": avg_val_total,   "region": avg_val_region,   "boundary": avg_val_boundary},
            "val_dice": avg_dice
        }
        history["epochs"].append(epoch_rec)
        METRICS_JSON.write_text(json.dumps(history, indent=2))

        # checkpoint
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_dice": avg_dice
            }, CHECKPOINT_PATH)
            print(f"  ✔ Saved new best model (Dice {avg_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no Val Dice improvement).")
                break

    print(f"Best Val Dice: {best_val_dice:.4f}")
    return CHECKPOINT_PATH

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
