#!/usr/bin/env python3
"""
Finetune auf:
  - sauberen Masken (GT)
  - harten Fällen

- lädt ein vortrainiertes Pseudo-Modell
- reduziert LR (default 3e-5)
- oversamplet harte Beispiele
- Early stopping & best checkpoint
"""

import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ───────────────────────────── CONFIG ─────────────────────────────
CLEAN_IMAGES = Path("/home/ansible/sarah/background_segmentation/dataset/images_cleaned")
CLEAN_MASKS  = Path("/home/ansible/sarah/background_segmentation/dataset/final_masks_cleaned")
HARD_IMAGES  = Path("/home/ansible/sarah/background_segmentation/dataset/images_hard")
HARD_MASKS   = Path("/home/ansible/sarah/background_segmentation/dataset/masks_hard")

INIT_CKPT = Path("/home/ansible/sarah/background_segmentation/checkpoints_pretrained/pseudo_model_all.pth")
OUT_CKPT  = Path("/home/ansible/sarah/background_segmentation/checkpoints_pretrained/pseudo_finetuned_clean_hard.pth")

IMG_SIZE = (512, 512)
LETTERBOX = False
SIDE_PADDING_RATIO = 0.1

EPOCHS = 20
BATCH_SIZE = 6
LR = 3e-5                # kleiner als beim Pretrain
ENCODER_LR_MULT = 0.3    # Encoder noch kleiner
WEIGHT_DECAY = 1e-4
PATIENCE = 8
VAL_FRACTION = 0.1
SEED = 1337

HARD_OVERSAMPLE_RATIO = 2.0
NUM_WORKERS = 4

# ───────────────────────────── UTILS ─────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA"); return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS"); return torch.device("mps")
    print("Using CPU"); return torch.device("cpu")

def letterbox_image_with_side_padding(image, padding_color=(0, 0, 0), side_padding_ratio=SIDE_PADDING_RATIO):
    image_np = np.array(image)
    orig_height, orig_width = image_np.shape[:2]

    side_padding = round(orig_width * side_padding_ratio)
    padded_width = orig_width + 2 * side_padding
    padded_height = orig_height

    padded_image = np.full((padded_height, padded_width, 3), padding_color, dtype=np.uint8)
    padded_image[:, side_padding:side_padding+orig_width] = image_np

    max_dim = max(padded_width, padded_height)
    letterboxed_image = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)
    x_offset = (max_dim - padded_width) // 2
    y_offset = (max_dim - padded_height) // 2
    letterboxed_image[y_offset:y_offset+padded_height, x_offset:x_offset+padded_width] = padded_image
    return letterboxed_image


# ───────────────────────────── DATASET ─────────────────────────────
class PairDataset(Dataset):
    IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    def __init__(self, images_root: Path, masks_root: Path, img_size=(512,512),
                 use_letterbox=False, side_padding_ratio=0.1, tag="clean"):
        self.images_root, self.masks_root = images_root, masks_root
        self.img_size, self.use_letterbox, self.side_padding_ratio = img_size, use_letterbox, side_padding_ratio
        self.samples = self._collect()
        print(f"[{tag}] {len(self.samples)} Paare gefunden.")

    def _collect(self):
        pairs = []
        for img in self.images_root.rglob("*"):
            if img.suffix.lower() not in self.IMG_EXTS: continue
            rel = img.relative_to(self.images_root)
            mask = (self.masks_root / rel).with_suffix(".png")
            if mask.exists(): pairs.append((img, mask))
        return pairs

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if LETTERBOX: img = letterbox_image_with_side_padding(img, self.side_padding_ratio)
        img  = cv2.resize(img,  self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        return img, mask

# ───────────────────────────── TRANSFORMS ─────────────────────────────
IMAGENET_MEAN, IMAGENET_STD = (0.485,0.456,0.406), (0.229,0.224,0.225)
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.25),
    A.HueSaturationValue(p=0.15),
    A.GaussNoise(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
val_tf = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

class WrappedSubset(Dataset):
    def __init__(self, base: PairDataset, indices: List[int], tf): self.base, self.indices, self.tf = base, indices, tf
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, mask = self.base[self.indices[i]]
        aug = self.tf(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)/255.0

# ───────────────────────────── MODEL ─────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w, self.smooth = bce_weight, smooth
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs*targets).sum((2,3))
        denom = probs.sum((2,3)) + targets.sum((2,3))
        dice = 1 - ((2*inter+self.smooth)/(denom+self.smooth)).mean()
        return self.w*bce + (1-self.w)*dice

def build_model():
    return smp.Unet("efficientnet-b0", encoder_weights="imagenet",
                    in_channels=3, classes=1, activation=None)

def dice_metric(logits, targets, smooth=1.0, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs*targets).sum((2,3))
    denom = probs.sum((2,3)) + targets.sum((2,3))
    return ((2*inter+smooth)/(denom+smooth+eps)).mean().item()

# ───────────────────────────── TRAIN ─────────────────────────────
def split_indices(n, val_fraction=VAL_FRACTION, seed=SEED):
    idxs = list(range(n)); random.Random(seed).shuffle(idxs)
    n_val = max(1, int(n*val_fraction))
    return idxs[n_val:], idxs[:n_val]

def train():
    set_seed(SEED)
    device = get_device(); use_amp = device.type=="cuda"

    ds_clean = PairDataset(CLEAN_IMAGES, CLEAN_MASKS, IMG_SIZE, LETTERBOX, SIDE_PADDING_RATIO, "clean")
    ds_hard  = PairDataset(HARD_IMAGES,  HARD_MASKS,  IMG_SIZE, LETTERBOX, SIDE_PADDING_RATIO, "hard")

    tr_c, va_c = split_indices(len(ds_clean)); tr_h, va_h = split_indices(len(ds_hard))
    train_ds = torch.utils.data.ConcatDataset([
        WrappedSubset(ds_clean, tr_c, train_tf),
        WrappedSubset(ds_hard,  tr_h, train_tf)
    ])
    val_ds = torch.utils.data.ConcatDataset([
        WrappedSubset(ds_clean, va_c, val_tf),
        WrappedSubset(ds_hard,  va_h, val_tf)
    ])

    weights = np.concatenate([np.ones(len(tr_c)), np.full(len(tr_h), HARD_OVERSAMPLE_RATIO)])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    criterion = BCEDiceLoss()
    if INIT_CKPT.exists():
        ckpt = torch.load(INIT_CKPT, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Geladenes Init: {INIT_CKPT}")

    opt = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LR*ENCODER_LR_MULT},
        {"params": model.decoder.parameters(), "lr": LR},
        {"params": model.segmentation_head.parameters(), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val, epochs_no_improve = -1.0, 0
    for epoch in range(EPOCHS):
        model.train(); run_loss=0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for data,target in pbar:
            data,target=data.to(device),target.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits=model(data); loss=criterion(logits,target)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss+=loss.item(); pbar.set_postfix({"loss":f"{loss.item():.4f}"})
        # val
        model.eval(); vloss=0; vdice=[]
        with torch.no_grad():
            for data,target in val_loader:
                data,target=data.to(device),target.to(device)
                logits=model(data); vloss+=criterion(logits,target).item()
                vdice.append(dice_metric(logits,target))
        avg_dice=sum(vdice)/len(vdice)
        print(f"Epoch {epoch+1}: val dice {avg_dice:.4f}")
        if avg_dice>best_val:
            best_val=avg_dice; epochs_no_improve=0
            torch.save({"epoch":epoch+1,"model_state_dict":model.state_dict(),"val_dice":best_val},OUT_CKPT)
            print(f" ✔ Neues bestes Modell → {OUT_CKPT}")
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=PATIENCE:
                print("Early stopping."); break
    print(f"Best Val Dice: {best_val:.4f}")

if __name__=="__main__":
    train()
