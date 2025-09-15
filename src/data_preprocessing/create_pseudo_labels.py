# eval_unet_smp.py
import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random
import json

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = Path("/home/sarah/Documents/background_segmentation/cart_segmentation/cart-segmentation-unet/finetuned_model_15.pth")

CHECKPOINTS = [
    Path("/home/sarah/Documents/background_segmentation/cart_segmentation/cart-segmentation-unet/finetuned_model_15.pth"),
    Path("/home/sarah/Documents/background_segmentation/cart_segmentation/cart-segmentation-unet/finetuned_model_14.pth"),
    Path("/home/sarah/Documents/background_segmentation/cart_segmentation/cart-segmentation-unet/finetuned_model_12.pth")    
]
RELEVANT_ROOT   = Path("/home/sarah/Documents/background_segmentation/head_and_shoulders_sarah")

OUT_DIR = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_mixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

overlay_dir = Path("/home/sarah/Documents/background_segmentation/preds_overlay_mixed")
IMG_SIZE = (512, 512)
SIDE_PADDING_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# replace the fixed THRESHOLD=... with:
thr_json = CHECKPOINT_PATH.with_suffix(".threshold.json")
try:
    THRESHOLD = float(json.load(open(thr_json))["threshold"])
except Exception:
    THRESHOLD = 0.5


# ---------------- UTILS ----------------
def letterbox_image_with_side_padding(image, padding_color=(0, 0, 0), side_padding_ratio=SIDE_PADDING_RATIO):
    image_np = np.array(image)
    orig_height, orig_width = image_np.shape[:2]

    side_padding = round(orig_width * side_padding_ratio)
    padded_width = orig_width + 2 * side_padding
    padded_height = orig_height

    padded_image = np.full((padded_height, padded_width, 3), padding_color, dtype=np.uint8)
    padded_image[:, side_padding:side_padding+orig_width] = image

    max_dim = max(padded_width, padded_height)
    letterboxed_image = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)

    x_offset = (max_dim - padded_width) // 2
    y_offset = (max_dim - padded_height) // 2
    letterboxed_image[y_offset:y_offset+padded_height, x_offset:x_offset+padded_width] = padded_image

    return letterboxed_image

# Preprocessing
transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ---------------- MODEL ----------------
def load_model(checkpoint_path):
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_ensemble(checkpoints, weights):
    models = [load_model(cp) for cp in checkpoints]
    weights = np.array(weights) / np.sum(weights)  # normalize
    return models, weights

# ---------------- INFERENCE + PLOT ----------------
def predict_and_plot(model, image_path, threshold=0.5):
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Letterbox like in training
    img_lb = letterbox_image_with_side_padding(img_rgb)

    # Prepare for model
    aug = transform(image=img_lb)
    img_t = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask_pred = (probs > threshold).astype(np.uint8)

    # Mask is already 512×512, overlay directly on letterboxed image
    mask_resized = cv2.resize(mask_pred, (img_lb.shape[1], img_lb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Get mask outline
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_img = img_lb.copy()
    cv2.drawContours(overlay_img, contours, -1, (255, 0, 0), thickness=2)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_img)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()

def clean_mask(pred01, min_area=1000):
    """
    Remove small connected components (artefacts) below `min_area` pixels.
    """
    # label connected components
    pred01 = (pred01.astype(np.uint8) > 0).astype(np.uint8)  # ensure {0,1} uint8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred01, connectivity=8)
    
    cleaned = np.zeros_like(pred01)
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    return cleaned

import math, csv

QC_CSV = OUT_DIR / "qc_scores.csv"
UNCERTAIN_DIR = OUT_DIR / "uncertain"; UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)

# thresholds: tune to your data after one pass
MIN_FG_AREA = 0.005   # 0.5% of image
MAX_FG_AREA = 0.60    # 60% of image
MIN_FG_CONF = 0.65
MAX_MEAN_ENTROPY = 0.35
MIN_TTA_IOU = 0.75
MIN_EDGE_HIT = 0.10

def entropy_map(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def tta_flip_iou(models, weights, rgb_lb, threshold):
    tens = transform(image=rgb_lb)["image"].unsqueeze(0).to(DEVICE)
    # normal
    probs = np.sum([
        torch.sigmoid(m(tens))[0,0].cpu().numpy() * w
        for m, w in zip(models, weights)
    ], axis=0)
    pred = (probs >= threshold).astype(np.uint8)
    # flip
    rgb_flip = np.ascontiguousarray(rgb_lb[:, ::-1, :])
    tens_f = transform(image=rgb_flip)["image"].unsqueeze(0).to(DEVICE)
    probs_f = np.sum([
        torch.sigmoid(m(tens_f))[0,0].cpu().numpy()[:, ::-1] * w
        for m, w in zip(models, weights)
    ], axis=0)
    pred_f = (probs_f >= threshold).astype(np.uint8)
    inter = np.logical_and(pred, pred_f).sum()
    union = np.logical_or(pred, pred_f).sum()
    return (inter / union) if union else 0.0


def edge_hit_rate(rgb_lb_512, pred01):
    # edges on gray
    gray = cv2.cvtColor(rgb_lb_512, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # boundary of mask
    cnts, _ = cv2.findContours(pred01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(pred01)
    cv2.drawContours(boundary, cnts, -1, 1, thickness=1)
    # how many boundary pixels overlap strong edges?
    hits = (edges > 0) & (boundary > 0)
    tot = (boundary > 0).sum()
    return (hits.sum() / tot) if tot else 0.0

def should_accept(fg_area, fg_conf, mean_entropy, tta_iou, edge_hit):
    if fg_area < MIN_FG_AREA or fg_area > MAX_FG_AREA: return False
    if fg_conf < MIN_FG_CONF: return False
    if mean_entropy > MAX_MEAN_ENTROPY: return False
    if tta_iou < MIN_TTA_IOU: return False
    if edge_hit < MIN_EDGE_HIT: return False
    return True

# init CSV
if not QC_CSV.exists():
    with open(QC_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","fg_area","fg_conf","mean_entropy","tta_iou","edge_hit","accepted","threshold"])

@torch.no_grad()

def predict_and_save(models, weights, image_path: Path, threshold: float = THRESHOLD):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"[WARN] cannot read {image_path}")
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb_lb = letterbox_image_with_side_padding(rgb)
    tens = transform(image=rgb_lb)["image"].unsqueeze(0).to(DEVICE)

    # --- Ensemble prediction ---
    probs_list = []
    for model, w in zip(models, weights):
        logits = model(tens)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        probs_list.append(probs * w)
    probs = np.sum(probs_list, axis=0)  # weighted average

    # threshold, cleanup, saving, QC ... (same as your current code)
    pred01   = (probs >= threshold).astype(np.uint8)
    min_area = int(IMG_SIZE[0] * IMG_SIZE[1] * 0.001)
    pred01   = clean_mask(pred01, min_area=min_area)

    rel_path  = image_path.relative_to(RELEVANT_ROOT)
    stem      = image_path.stem

    # ---- QC metrics (you already defined these helpers) ----
    rgb_lb_512 = cv2.resize(rgb_lb, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    fg_area   = float(pred01.mean())
    fg_conf   = float(np.median(np.abs(probs - 0.5) * 2.0))
    mean_ent  = float(entropy_map(probs).mean())
    tta_iou = float(tta_flip_iou(models, weights, rgb_lb, threshold))
    edge_hit  = float(edge_hit_rate(rgb_lb_512, pred01))
    accepted  = should_accept(fg_area, fg_conf, mean_ent, tta_iou, edge_hit)

    # base dir by acceptance
    base_dir = OUT_DIR / rel_path.parent if accepted else (UNCERTAIN_DIR / rel_path.parent)
    base_dir.mkdir(parents=True, exist_ok=True)

    # save hard mask + soft probs
    cv2.imwrite(str(base_dir / f"{stem}.png"), (pred01 * 255).astype(np.uint8))
    np.savez_compressed(base_dir / f"{stem}_prob.npz", prob=probs.astype(np.float16))

    # overlay for quick QC
    overlay = rgb_lb_512.copy()
    cnts, _ = cv2.findContours(pred01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)
    cv2.imwrite(str(overlay_dir / f"{stem}_overlay_512.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # log QC (you already created QC_CSV earlier)
    with open(QC_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([str(rel_path), f"{fg_area:.4f}", f"{fg_conf:.2f}", f"{mean_ent:.3f}",
                    f"{tta_iou:.2f}", f"{edge_hit:.2f}", int(accepted), f"{threshold:.2f}"])

    return accepted

MAX_MASKS = 100000
counter = 0
MAX_PER_BUCKET = 50
bucket_counts = {}

# ---------------- MAIN ----------------
if __name__ == "__main__":
    weights = [0.5, 0.25, 0.25]  # match checkpoints
    models, weights = load_ensemble(CHECKPOINTS, weights)

    image_paths = sorted(RELEVANT_ROOT.rglob("*.jpeg"))
    print(f"Found {len(image_paths)} images, will generate up to {MAX_MASKS} masks.")
    random.shuffle(image_paths)

    for img_path in image_paths:
        if counter >= MAX_MASKS:
            break
        bucket = img_path.parent.name
        if bucket_counts.get(bucket, 0) >= MAX_PER_BUCKET:
            continue
        ok = predict_and_save(models, weights, img_path, THRESHOLD)
        if ok:
            counter += 1
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
