# fast_pseudo_labels.py
import cv2
import torch
torch.backends.cudnn.benchmark = True
import numpy as np
from pathlib import Path
import csv, json, random
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------------- CONFIG ----------------
CHECKPOINTS = [
    Path("~/sarah/background_segmentation/checkpoints_pretrained/finetuned_model_15.pth").expanduser(),
    Path("~/sarah/background_segmentation/checkpoints_pretrained/finetuned_model_14.pth").expanduser(),
    Path("~/sarah/background_segmentation/checkpoints_pretrained/finetuned_model_12.pth").expanduser(),
]
ENSEMBLE_WEIGHTS = [0.5, 0.25, 0.25]  # must match CHECKPOINTS length

RELEVANT_ROOT = Path("/opt/whizcart/shared/carrefour_classes/images/head_and_shoulders_sub_sarah").expanduser()
OUT_DIR = Path("~/sarah/background_segmentation/dataset/pseudo_masks_head_and shoulders").expanduser()
UNCERTAIN_DIR = OUT_DIR / "uncertain"
OVERLAY_DIR = Path("~/sarah/background_segmentation/preds_overlay_mixed").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (512, 512)
SIDE_PADDING_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speed / I/O toggles
BATCH_SIZE = 16          # increase if you have more VRAM
SAVE_OVERLAYS = False    # overlays cost I/O and CPU
SAVE_SOFT_PROBS = False  # .npz per image; switch off for max speed
ENABLE_TTA_IOU = False   # very expensive; off = fastest
MAX_MASKS = 100000
MAX_PER_BUCKET = 50

# Input extensions
EXTS = ("*.jpeg", "*.jpg", "*.png")

# Threshold: use a fixed 0.5 for ensemble (robust & fast)
THRESHOLD = 0.5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# QC thresholds
MIN_FG_AREA    = 0.005  # 0.5%
MAX_FG_AREA    = 0.60   # 60%
MIN_FG_CONF    = 0.65
MAX_MEAN_ENT   = 0.35
MIN_TTA_IOU    = 0.75
MIN_EDGE_HIT   = 0.10

# ---------------- UTILS ----------------
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

transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# def clean_mask(pred01, min_area=1000):
#     pred01 = (pred01.astype(np.uint8) > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred01, connectivity=8)
#     cleaned = np.zeros_like(pred01)
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             cleaned[labels == i] = 1
#     return cleaned

def entropy_map(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def edge_hit_rate(rgb_lb_512, pred01):
    gray = cv2.cvtColor(rgb_lb_512, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(pred01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(pred01)
    cv2.drawContours(boundary, cnts, -1, 1, thickness=1)
    hits = (edges > 0) & (boundary > 0)
    tot = (boundary > 0).sum()
    return (hits.sum() / tot) if tot else 0.0

def should_accept(fg_area, fg_conf, mean_entropy, tta_iou, edge_hit):
    if fg_area < MIN_FG_AREA or fg_area > MAX_FG_AREA: return False
    if fg_conf < MIN_FG_CONF: return False
    if mean_entropy > MAX_MEAN_ENT: return False
    if ENABLE_TTA_IOU and (tta_iou < MIN_TTA_IOU): return False
    if edge_hit < MIN_EDGE_HIT: return False
    return True

# ---------------- MODEL ----------------
def load_model(checkpoint_path: Path):
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
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)  # PyTorch 2.x
        except Exception:
            pass
    return model

def load_ensemble(checkpoints: List[Path], weights: List[float]):
    models = [load_model(cp) for cp in checkpoints]
    w = np.array(weights, dtype=np.float32)
    w = (w / w.sum()).tolist()
    return models, w

# ---------------- DATASET ----------------
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, paths: List[Path]):
        self.root = root
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p: Path = self.paths[idx]
        bgr = cv2.imread(str(p))
        if bgr is None:
            # return a dummy sample; caller should skip None
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_lb = letterbox_image_with_side_padding(rgb)
        tens = transform(image=rgb_lb)["image"]  # CHW tensor (CPU)
        rel_path = p.relative_to(self.root)
        stem = p.stem
        bucket = p.parent.name
        # also keep a 512×512 RGB for overlay/edge metrics
        rgb_lb_512 = cv2.resize(rgb_lb, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        return tens, rgb_lb, rgb_lb_512, rel_path, stem, bucket

# ---------------- INFERENCE HELPERS ----------------
def ensemble_forward(models, weights, tens):
    # tens: [B,3,H,W] on DEVICE
    with torch.inference_mode():
        if DEVICE.type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            out_sum = None
            for m, w in zip(models, weights):
                logits = m(tens)                 # [B,1,H,W]
                p = torch.sigmoid(logits)[:, 0]  # [B,H,W]
                out_sum = p.mul_(w) if out_sum is None else out_sum.add_(p, alpha=w)
    return out_sum  # [B,H,W] float16/32 (AMP cast)

def tta_iou_for_batch(models, weights, rgb_lb_batch: List[np.ndarray], threshold: float):
    # This is optional and slow; do only if ENABLE_TTA_IOU=True
    # Run normal + flipped batch and compute IoU per sample
    tens = torch.stack([transform(image=x)["image"] for x in rgb_lb_batch], dim=0).to(DEVICE, non_blocking=True)
    probs = ensemble_forward(models, weights, tens).float().cpu().numpy()
    rgb_flip = [np.ascontiguousarray(x[:, ::-1, :]) for x in rgb_lb_batch]
    tens_f = torch.stack([transform(image=x)["image"] for x in rgb_flip], dim=0).to(DEVICE, non_blocking=True)
    probs_f = ensemble_forward(models, weights, tens_f).float().cpu().numpy()
    probs_f = probs_f[:, :, ::-1]
    ious = []
    for p, pf in zip(probs, probs_f):
        a = (p >= threshold).astype(np.uint8)
        b = (pf >= threshold).astype(np.uint8)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious.append((inter / union) if union else 0.0)
    return ious

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Collect image files
    image_paths: List[Path] = []
    for ext in EXTS:
        image_paths += list(RELEVANT_ROOT.rglob(ext))
    image_paths = sorted(image_paths)
    random.shuffle(image_paths)
    print(f"Found {len(image_paths)} images.")

    # Load ensemble
    models, weights = load_ensemble(CHECKPOINTS, ENSEMBLE_WEIGHTS)
    print(f"Loaded {len(models)} models with weights {weights}")

    # CSV init
    QC_CSV = OUT_DIR / "qc_scores.csv"
    if not QC_CSV.exists():
        with open(QC_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image","fg_area","fg_conf","mean_entropy","tta_iou","edge_hit","accepted","threshold"])

    # DataLoader (num_workers>0 prefetches & augments on CPU)
    ds = ImageDataset(RELEVANT_ROOT, image_paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=lambda batch: [b for b in batch if b is not None]
    )

    accepted_total = 0
    bucket_counts = {}

    for batch in loader:
        if accepted_total >= MAX_MASKS:
            break
        if len(batch) == 0:
            continue

        # Unpack batch
        tens_cpu, rgb_lb_list, rgb_lb_512_list, rel_paths, stems, buckets = zip(*batch)
        tens = torch.stack(tens_cpu, dim=0).to(DEVICE, non_blocking=True)  # [B,3,512,512]

        # Forward (ensemble + AMP)
        probs_t = ensemble_forward(models, weights, tens)
        probs_np = probs_t.float().cpu().numpy()  # [B,H,W]

        # Optional TTA IoU for QC (slow)
        if ENABLE_TTA_IOU:
            tta_ious = tta_iou_for_batch(models, weights, list(rgb_lb_list), THRESHOLD)
        else:
            tta_ious = [1.0] * len(probs_np)  # neutral high score to not block acceptance

        # Process each item in the batch
        for probs, rgb_lb_512, rel_p, stem, bucket, tta_iou_val in zip(
            probs_np, rgb_lb_512_list, rel_paths, stems, buckets, tta_ious
        ):
            if accepted_total >= MAX_MASKS:
                break

            # Threshold + cleanup
            pred01 = (probs >= THRESHOLD).astype(np.uint8)
            min_area = int(IMG_SIZE[0] * IMG_SIZE[1] * 0.001)
            # pred01 = clean_mask(pred01, min_area=min_area)

            # QC metrics
            fg_area  = float(pred01.mean())
            fg_conf  = float(np.median(np.abs(probs - 0.5) * 2.0))
            mean_ent = float(entropy_map(probs).mean())
            edge_hit = float(edge_hit_rate(rgb_lb_512, pred01))
            accepted = should_accept(fg_area, fg_conf, mean_ent, tta_iou_val, edge_hit)

            # Enforce per-bucket cap only if accepted
            if accepted and bucket_counts.get(bucket, 0) >= MAX_PER_BUCKET:
                accepted = False

            # Base dir mirrors input structure
            base_dir = OUT_DIR / rel_p.parent if accepted else (UNCERTAIN_DIR / rel_p.parent)
            base_dir.mkdir(parents=True, exist_ok=True)

            # Save mask
            cv2.imwrite(str(base_dir / f"{stem}.png"), (pred01 * 255).astype(np.uint8))

            # Optional: save soft probs
            if SAVE_SOFT_PROBS:
                np.savez_compressed(base_dir / f"{stem}_prob.npz", prob=probs.astype(np.float16))

            # Optional: overlay
            if SAVE_OVERLAYS:
                overlay = rgb_lb_512.copy()
                cnts, _ = cv2.findContours(pred01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)
                cv2.imwrite(str(OVERLAY_DIR / f"{stem}_overlay_512.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Log QC
            with open(QC_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([str(rel_p), f"{fg_area:.4f}", f"{fg_conf:.2f}", f"{mean_ent:.3f}",
                            f"{tta_iou_val:.2f}", f"{edge_hit:.2f}", int(accepted), f"{THRESHOLD:.2f}"])

            # Update counters
            if accepted:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                accepted_total += 1

    print(f"✅ Done. Accepted {accepted_total} masks.")
