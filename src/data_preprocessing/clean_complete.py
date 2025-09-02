#!/usr/bin/env python3
"""
Unified mask cleaner (hardcoded config)
- Removes artifacts (keeps only the largest connected component)
- Removes holes inside the main area
- Removes any mask parts that are inside the padding region
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

#  gemuese_netz
# ----------------------- HARD CODED CONFIG -----------------------
IMAGES_ROOT = Path("~/opt/whizcart/shared/carrefour_classes/images/merci").expanduser()
MASKS_ROOT  = Path("~/sarah/background_segmentation/dataset/pseudo_masks_mixed_merci").expanduser()
OUT_ROOT    = Path("~/sarah/background_segmentation/dataset/mixed_pseudo_clean").expanduser()

# IMAGES_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_224")
# MASKS_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/images_pseudo_224")
# OUT_ROOT    = Path("/home/sarah/Documents/background_segmentation/dataset/test_pseudo_masks_224")

SIDE_PADDING_RATIO = 0.1   # horizontal padding fraction
OPEN_KS  = 3               # morph open kernel size
CLOSE_KS = 7               # morph close kernel size
HOLE_FRAC = 0.004          # max hole area fraction to fill

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
MASK_EXTS  = {".png",".jpg",".jpeg"}
SKIP_EXISTING = False

# ----------------------- geometry helpers -----------------------

def calculate_content_boundaries(orig_w, orig_h, final_w, final_h, side_padding_ratio):
    side_padding = round(orig_w * side_padding_ratio)
    padded_w = orig_w + 2 * side_padding
    padded_h = orig_h

    max_dim = max(padded_w, padded_h)
    x_offset = (max_dim - padded_w) // 2
    y_offset = (max_dim - padded_h) // 2

    x0 = x_offset + side_padding
    y0 = y_offset
    x1 = x0 + orig_w
    y1 = y0 + orig_h

    sx = final_w / max_dim
    sy = final_h / max_dim
    X0 = int(round(x0 * sx)); Y0 = int(round(y0 * sy))
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))

    X0 = max(0, min(final_w, X0)); X1 = max(0, min(final_w, X1))
    Y0 = max(0, min(final_h, Y0)); Y1 = max(0, min(final_h, Y1))

    return X0, Y0, X1, Y1

def cut_mask_padding(mask, orig_w, orig_h, side_padding_ratio):
    H, W = mask.shape[:2]
    x0, y0, x1, y1 = calculate_content_boundaries(orig_w, orig_h, W, H, side_padding_ratio)
    content_mask = np.zeros((H, W), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        content_mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(mask, content_mask)

# ----------------------- cleaning helpers -----------------------

def fill_small_holes(fg_mask_255, max_hole_area_frac=0.004):
    m = (fg_mask_255 > 127).astype(np.uint8) * 255
    H, W = m.shape
    max_area = int(H * W * max_hole_area_frac)
    inv = 255 - m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = m.copy()
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0) or (y == 0) or (x + w == W) or (y + h == H)
        if touches_border:
            continue
        if area <= max_area:
            filled[labels == i] = 255
    return filled

def keep_largest_component(mask255):
    m = (mask255 > 127).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m * 255
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_idx).astype(np.uint8) * 255

def post_clean(binary_mask, open_ks=3, close_ks=7, hole_frac=0.004):
    m = (binary_mask > 127).astype(np.uint8) * 255
    if open_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = keep_largest_component(m)
    if close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = fill_small_holes(m, max_hole_area_frac=hole_frac)
    return m

# ----------------------- IO helpers -----------------------

def get_original_dimensions_cv2(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return w, h

def first_existing_with_exts(base_path: Path, exts):
    for ext in exts:
        p = base_path.with_suffix(ext)
        if p.exists():
            return p
    return None

# ----------------------- main pipeline -----------------------

def clean_one_mask(mask_path: Path, img_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    ow, oh = get_original_dimensions_cv2(img_path)
    if ow is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    m = cut_mask_padding(mask, ow, oh, SIDE_PADDING_RATIO)
    m = post_clean(m, open_ks=OPEN_KS, close_ks=CLOSE_KS, hole_frac=HOLE_FRAC)
    return m

def process_all():
    images = [p for p in IMAGES_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    processed = skipped = errors = 0
    for img_file in tqdm(images, desc="Cleaning masks"):
        try:
            rel = img_file.relative_to(IMAGES_ROOT)
            in_mask = first_existing_with_exts(MASKS_ROOT / rel, MASK_EXTS)
            if in_mask is None:
                continue
            out_mask = (OUT_ROOT / rel).with_suffix(".png")
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            if SKIP_EXISTING and out_mask.exists():
                skipped += 1
                continue
            cleaned = clean_one_mask(in_mask, img_file)
            cv2.imwrite(str(out_mask), cleaned)
            processed += 1
        except Exception as e:
            print(f"⚠️ Error on {img_file}: {e}")
            errors += 1
    print("\n✅ Done.")
    print(f"   Processed: {processed}")
    print(f"   Skipped:   {skipped}")
    print(f"   Errors:    {errors}")
    print(f"   Output:    {OUT_ROOT}")

def main():
    process_all()

if __name__ == "__main__":
    main()