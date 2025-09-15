import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
MASKS_DIR   = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_224")          # input dir
OUTPUT_DIR  = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_224_cleaned")  # output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# file extensions to look for
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def clean_mask_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    Assumes mask is binary {0, 255} or {0, 1}.
    Returns a cleaned binary mask.
    """
    # normalize to {0,1}
    mask_bin = (mask > 0).astype(np.uint8)

    # connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    if num_labels <= 1:
        return mask_bin  # nothing found except background

    # find largest component (skip background at index 0)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = (labels == largest_idx).astype(np.uint8)

    return cleaned * 255  # back to {0,255}

# ---------------- MAIN ----------------
def main():
    mask_paths = [p for p in MASKS_DIR.rglob("*") if p.suffix.lower() in EXTS]

    for mask_path in tqdm(mask_paths, desc="Cleaning masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Could not read {mask_path}")
            continue

        cleaned = clean_mask_largest_component(mask)

        out_path = OUTPUT_DIR / mask_path.relative_to(MASKS_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cleaned)

    print(f"✅ Done. Cleaned masks saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
