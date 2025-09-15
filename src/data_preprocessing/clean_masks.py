import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
MASKS_DIR = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks")
OUTPUT_DIR = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_mask(mask):
    # Ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Fill small holes using flood fill
    filled = binary.copy()
    h, w = binary.shape
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask_ff, (0, 0), 255)
    inv_filled = cv2.bitwise_not(filled)
    clean = cv2.bitwise_or(binary, inv_filled)

    # Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_labels <= 1:
        return clean  # No objects found or only background

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    return largest_component

# === Main Loop ===
mask_files = list(MASKS_DIR.rglob("*.png"))
print(f"🔍 Found {len(mask_files)} mask files")

for mask_path in tqdm(mask_files, desc="Cleaning Masks"):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Could not read: {mask_path}")
        continue

    cleaned = clean_mask(mask)

    out_path = OUTPUT_DIR / mask_path.relative_to(MASKS_DIR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cleaned)

print("✅ All masks cleaned and saved.")
