# finetune_pseudo.py  — strict 224 fine-tuning with hard/cleaned/pseudo mix
import os, json, random
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------------- PATHS (ANPASSEN) ----------------
CHECKPOINT_IN  = Path("~/sarah/background_segmentation/checkpoints_pretrained/finetuned_model_15.pth").expanduser()
CHECKPOINT_OUT = CHECKPOINT_IN.with_name("224_finetuned_model_15.pth")

# Hard cases (alle 430)
HARD_IMG_ROOT   = Path("~/sarah/background_segmentation/dataset/images_hard").expanduser()
HARD_MASK_ROOT  = Path("~/sarah/background_segmentation/dataset/masks_hard").expanduser()

# Cleaned subset (300–600 aus den 1500; nimm nur den Teil, den du willst)
CLEAN_IMG_ROOT  = Path("~/sarah/background_segmentation/dataset/images_cleaned").expanduser()
CLEAN_MASK_ROOT = Path("~/sarah/background_segmentation/dataset/final_masks_cleaned").expanduser()

# Pseudo akzeptiert (1000–1500)
PSEUDO_IMG_ROOT  = Path("~/sarah/background_segmentation/dataset/images_pseudo_224").expanduser()
PSEUDO_MASK_ROOT = Path("~/sarah/background_segmentation/dataset/pseudo_masks_224").expanduser()

# ---------------- TRAINING KNOBS ----------------
TARGET_SIZE = 224
SIDE_PADDING_RATIO = 0.10
USE_IMAGENET_NORM = True

BATCH_SIZE = 20
EPOCHS = 20
PATIENCE = 5

# Encoder warmup
WARMUP_EPOCHS = 2
HEAD_LR = 1e-4
ENCODER_LR = 3e-5
WEIGHT_DECAY = 1e-4

USE_AMP = True
NUM_WORKERS = 4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sampling-Anteile (ungefähr pro Batch)
RATIO_HARD   = 0.50
RATIO_CLEAN  = 0.25
RATIO_PSEUDO = 0.25

# Loss-Gewichte (Pseudo-Curriculum)
LOSS_W_HARD = 1.0
LOSS_W_CLEAN = 1.0
LOSS_W_PSEUDO_START = 0.4
LOSS_W_PSEUDO_END   = 0.6
PSEUDO_RAMP_EPOCHS  = 4

# Val-Anteil aus den Hard-Cases (224-Validierung)
VAL_FRAC = 0.12

# ---------------- UTILS ----------------
def set_seeds(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def letterbox_sidepad(image, padding_color=(0,0,0), side_ratio=SIDE_PADDING_RATIO):
    h, w = image.shape[:2]
    sp = round(w * side_ratio)
    padded = np.full((h, w+2*sp, 3), padding_color, dtype=np.uint8)
    padded[:, sp:sp+w] = image
    s = max(h, w+2*sp)
    canvas = np.full((s, s, 3), padding_color, dtype=np.uint8)
    x0 = (s - (w+2*sp))//2
    y0 = (s - h)//2
    canvas[y0:y0+h, x0:x0+w+2*sp] = padded
    return canvas

class Fixed224Transform:
    def __init__(self, size=TARGET_SIZE, use_imagenet_norm=True, train=True):
        if use_imagenet_norm:
            self.mean=(0.485,0.456,0.406); self.std=(0.229,0.224,0.225)
        else:
            self.mean=(0,0,0); self.std=(1,1,1)
        aug = [A.Resize(size, size, interpolation=cv2.INTER_LINEAR)]
        if train:
            aug += [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.12, rotate_limit=5,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.20),
                A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02, p=0.15),
            ]
        aug += [A.Normalize(mean=self.mean, std=self.std), ToTensorV2()]
        self.aug = A.Compose(aug)

    def __call__(self, image, mask):
        res = self.aug(image=image, mask=mask)
        x = res["image"]
        y = res["mask"].unsqueeze(0).float()
        return x, y

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

class PairDataset(Dataset):
    """Spiegelt Dateistruktur: gleiche rel. Pfade unter img_root & mask_root."""
    def __init__(self, img_root:Path, mask_root:Path, transform:Fixed224Transform):
        self.img_root = Path(img_root); self.mask_root = Path(mask_root); self.transform = transform
        imgs = [p for p in self.img_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        pairs = []
        for ip in imgs:
            rel = ip.relative_to(self.img_root)
            for e in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
                mp = (self.mask_root / rel).with_suffix(e)
                if mp.exists():
                    pairs.append((ip, mp)); break
        if not pairs:
            raise RuntimeError(f"No pairs found under {img_root} vs {mask_root}")
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img_bgr is None: raise RuntimeError(f"Cannot read image: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb = letterbox_sidepad(img_rgb)

        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None: raise RuntimeError(f"Cannot read mask: {mp}")
        m01 = (m > 127).astype(np.float32)

        x, y = self.transform(image=img_lb, mask=m01)
        return x, y

@dataclass
class SourceSpec:
    name: str
    ds: Dataset
    ratio: float
    loss_weight: float

def build_sources():
    tfm_train = Fixed224Transform(TARGET_SIZE, USE_IMAGENET_NORM, train=True)
    hard = PairDataset(HARD_IMG_ROOT,  HARD_MASK_ROOT,  tfm_train)
    clean= PairDataset(CLEAN_IMG_ROOT, CLEAN_MASK_ROOT, tfm_train)
    pseudo=PairDataset(PSEUDO_IMG_ROOT,PSEUDO_MASK_ROOT,tfm_train)
    return [
        SourceSpec("hard",   hard,  RATIO_HARD,   LOSS_W_HARD),
        SourceSpec("clean",  clean, RATIO_CLEAN,  LOSS_W_CLEAN),
        SourceSpec("pseudo", pseudo,RATIO_PSEUDO, LOSS_W_PSEUDO_START),
    ]

def split_hard_for_val(hard_ds:Dataset, val_frac:float, seed:int=SEED):
    n = len(hard_ds); n_val = max(1, int(val_frac*n))
    rng = np.random.RandomState(seed); idx = rng.permutation(n)
    val_idx = set(idx[:n_val].tolist())
    train_idx = [i for i in range(n) if i not in val_idx]
    return torch.utils.data.Subset(hard_ds, train_idx), torch.utils.data.Subset(hard_ds, list(val_idx))

class MixedDataset(Dataset):
    """Alle Quellen zusammen; WeightedRandomSampler steuert das Verhältnis."""
    def __init__(self, sources:list[SourceSpec]):
        self.datasets = [s.ds for s in sources]
        self.source_names = [s.name for s in sources]
        self.source_loss_w = [s.loss_weight for s in sources]
        self.ratios = [s.ratio for s in sources]
        self.mapping = []
        for si, ds in enumerate(self.datasets):
            for li in range(len(ds)):
                self.mapping.append((si, li, si))
        self.total_len = len(self.mapping)
        # pro Sample: ratio[source] / len(source_ds)
        counts = [len(d) for d in self.datasets]
        self.sample_weights = []
        offset = 0
        for si, n in enumerate(counts):
            w = (self.ratios[si] / max(1,n))
            self.sample_weights += [w]*n

    def __len__(self): return self.total_len

    def __getitem__(self, idx):
        si, li, src = self.mapping[idx]
        x, y = self.datasets[si][li]
        return x, y, torch.tensor(src, dtype=torch.long)

# ---------------- MODEL/LOSS ----------------
def build_model():
    m = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)
    return m

class BCEDiceLossPerSample(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_weight = bce_weight; self.smooth = smooth
    def forward(self, logits, targets):
        # logits/targets: [B,1,H,W]
        bce_map = self.bce(logits, targets)         # [B,1,H,W]
        bce = bce_map.mean(dim=(1,2,3))             # [B]
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1,2,3))
        den = (probs + targets).sum(dim=(1,2,3))
        dice = 1 - (2*inter + self.smooth) / (den + self.smooth)
        return 0.5 * bce + 0.5 * dice

@torch.no_grad()
def dice_iou_at_t(logits, targets, t=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > t).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    den = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    union = den - inter
    dice = (2*inter + eps) / (den + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice.mean().item()), float(iou.mean().item())

@torch.no_grad()
def sweep_best_threshold(model, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, steps=19)
    model.eval()
    metrics=[]
    for t in thresholds:
        dices, ious = [], []
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            d, i = dice_iou_at_t(logits, y, t=float(t))
            dices.append(d); ious.append(i)
        metrics.append((float(t), float(np.mean(dices)), float(np.mean(ious))))
    return max(metrics, key=lambda z: z[1])

def build_param_groups_for_smp(model, encoder_lr, head_lr, weight_decay=0.0):
    enc, head = [], []
    if hasattr(model, "encoder"):
        enc = [p for p in model.encoder.parameters() if p.requires_grad]
    if hasattr(model, "decoder"):
        head += list(model.decoder.parameters())
    if hasattr(model, "segmentation_head"):
        head += list(model.segmentation_head.parameters())
    seen=set(); head=[p for p in head if not (id(p) in seen or seen.add(id(p)))]
    return [
        {"params": enc, "lr": float(encoder_lr), "weight_decay": weight_decay},
        {"params": head, "lr": float(head_lr),   "weight_decay": weight_decay},
    ]

class EncoderWarmupController:
    def __init__(self, model, warmup_epochs:int):
        self.model = model; self.warm = max(int(warmup_epochs), 0)
        if self.warm>0 and hasattr(model, "encoder"):
            for p in model.encoder.parameters(): p.requires_grad = False
    def maybe_unfreeze(self, current_epoch:int):
        if self.warm==0 or not hasattr(self.model, "encoder"): return
        if current_epoch == self.warm:
            for p in self.model.encoder.parameters(): p.requires_grad = True

# ---------------- LOADERS ----------------
def make_train_val_loaders():
    sources = build_sources()
    # split nur die Hard-Cases für Val (224)
    hard_ds = sources[0].ds
    n = len(hard_ds); n_val = max(1, int(VAL_FRAC*n))
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    val_idx = set(idx[:n_val].tolist())
    train_idx = [i for i in range(n) if i not in val_idx]
    hard_train = torch.utils.data.Subset(hard_ds, train_idx)
    hard_val   = torch.utils.data.Subset(hard_ds, list(val_idx))
    # ersetze hard in sources durch hard_train
    sources[0] = SourceSpec(sources[0].name, hard_train, sources[0].ratio, sources[0].loss_weight)

    mix = MixedDataset(sources)
    weights = torch.as_tensor(mix.sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(mix), replacement=True)

    train_loader = DataLoader(mix, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    val_loader = DataLoader(hard_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    return train_loader, val_loader

# ---------------- TRAIN ----------------
def finetune_fixed224_mix():
    set_seeds(SEED)
    train_loader, val_loader = make_train_val_loaders()

    model = build_model()
    ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)

    crit = BCEDiceLossPerSample(bce_weight=0.5, smooth=1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    warm = EncoderWarmupController(model, WARMUP_EPOCHS)

    optimizer = optim.AdamW(build_param_groups_for_smp(model, ENCODER_LR, HEAD_LR, WEIGHT_DECAY), lr=HEAD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_dice = -1.0
    patience = 0

    base_w = torch.tensor([LOSS_W_HARD, LOSS_W_CLEAN, 1.0], device=DEVICE)

    for epoch in range(EPOCHS):
        # curriculum für pseudo
        alpha = min(1.0, (epoch+1)/max(1,PSEUDO_RAMP_EPOCHS))
        w_pseudo = LOSS_W_PSEUDO_START + alpha*(LOSS_W_PSEUDO_END - LOSS_W_PSEUDO_START)

        model.train()
        warm.maybe_unfreeze(epoch)

        run_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train 224]")
        for xb, yb, src in pbar:
            xb, yb, src = xb.to(DEVICE), yb.to(DEVICE), src.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(xb)
                per_sample = crit(logits, yb)    # [B]
                weights = base_w.clone(); weights[2] = w_pseudo
                w = weights[src]                 # [B]
                loss = (per_sample * w).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "w_pseudo": f"{w_pseudo:.2f}"})

        scheduler.step()

        # ---- VALIDIERUNG (strict 224) ----
        model.eval()
        vloss, dices, ious = 0.0, [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                per_sample = crit(logits, yb)
                vloss += per_sample.mean().item()
                d, i = dice_iou_at_t(logits, yb, t=0.5)
                dices.append(d); ious.append(i)

        avg_tr = run_loss / max(1, len(train_loader))
        avg_vl = vloss   / max(1, len(val_loader))
        avg_d  = float(np.mean(dices)); avg_i = float(np.mean(ious))
        print(f"[FT-224] Epoch {epoch+1}/{EPOCHS} | Train {avg_tr:.4f} | Val {avg_vl:.4f} | Dice {avg_d:.4f} | IoU {avg_i:.4f} | w_pseudo {w_pseudo:.2f}")

        if avg_d > best_val_dice:
            best_val_dice = avg_d; patience = 0
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_d,
                "val_iou": avg_i
            }, CHECKPOINT_OUT)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[FT-224] Early stopping after {PATIENCE} epochs without Val Dice improvement.")
                break

    # ---- bestes Modell laden & threshold sweep (Val @224) ----
    best = torch.load(CHECKPOINT_OUT, map_location=DEVICE)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    t, d, i = sweep_best_threshold(model, val_loader, DEVICE)
    print(f"[FT-224] Best threshold on val: t={t:.2f} | Dice={d:.4f} | IoU={i:.4f}")
    thr_json = CHECKPOINT_OUT.with_suffix(".threshold.json")
    with open(thr_json, "w") as f:
        json.dump({"threshold": t, "dice": d, "iou": i}, f, indent=2)
    print(f"[FT-224] Saved: {CHECKPOINT_OUT} and {thr_json}")
    return CHECKPOINT_OUT, thr_json

if __name__ == "__main__":
    set_seeds(SEED)
    finetune_fixed224_mix()
