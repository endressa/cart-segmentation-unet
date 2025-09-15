import shutil
from pathlib import Path
from typing import Optional

# Paths
DATASET_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset")
MASKS_CLEANED = DATASET_ROOT / "final_masks_cleaned"
MASKS_PSEUDO  = DATASET_ROOT / "pseudo_masks_15_best"

IMAGES_CLEANED = DATASET_ROOT / "images_cleaned"
IMAGES_PSEUDO  = DATASET_ROOT / "images_pseudo_224"

IMAGES_HARD = DATASET_ROOT / "images_hard"
MASKS_HARD = DATASET_ROOT / "masks_hard"
IMAGES_CLEANED.mkdir(parents=True, exist_ok=True)
IMAGES_PSEUDO.mkdir(parents=True, exist_ok=True)
IMAGES_HARD.mkdir(parents=True, exist_ok=True)

# Source image roots
IMAGES_CLEANED_SRC = Path("/home/sarah/Documents/background_segmentation/v2_expansion")
IMAGES_PSEUDO_SRC  = Path("/home/sarah/Documents/background_segmentation/head_and_shoulders_sarah/head_and_shoulders_sub_sarah/raw")
IMAGES_HARD_SRC = Path("/home/sarah/Documents/background_segmentation/whole_datasets/ariel_sarah/raw")

# Valid (lowercased) image extensions
EXTS = {".jpg", ".jpeg", ".png"}

def find_image(stem: str, rel_dir: Path, src_root: Path) -> Optional[Path]:
    """
    rel_dir is the FULL nested path under the mask root
    (e.g., 'store_xxx/session_yyy'), so keep it intact.
    """
    cand_dir = src_root / rel_dir  # preserve store/session structure
    if not cand_dir.exists():
        return None

    # Fast exact-match by common extensions (case-insensitive)
    for ext in EXTS:
        p = cand_dir / f"{stem}{ext}"
        if p.exists():
            return p
        # try uppercase variant too
        if (cand_dir / f"{stem}{ext.upper()}").exists():
            return cand_dir / f"{stem}{ext.upper()}"

    # Fallback: any file with same stem and allowed suffix
    for p in cand_dir.glob(stem + ".*"):
        if p.suffix.lower() in EXTS:
            return p

    return None

def copy_images(mask_root: Path, image_root: Path, src_root: Path):
    mask_files = list(mask_root.rglob("*.png"))
    print(f"Found {len(mask_files)} masks in {mask_root}")

    copied = 0
    missing = 0
    for mask_path in mask_files:
        # rel_dir like 'store_xxx/session_yyy'
        rel_path = mask_path.relative_to(mask_root)
        rel_dir  = rel_path.parent
        stem     = mask_path.stem

        img_path = find_image(stem, rel_dir, src_root)
        if img_path:
            out_path = image_root / rel_dir / (stem + img_path.suffix.lower())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_path)
            copied += 1
        else:
            print(f"⚠️ No image for mask: {rel_dir}/{stem}.png")
            missing += 1

    print(f"✅ Copied {copied} images into {image_root} | ❌ Missing: {missing}")

# Run for pseudo (and cleaned if you want)
# copy_images(MASKS_CLEANED, IMAGES_CLEANED, IMAGES_CLEANED_SRC)
copy_images(MASKS_PSEUDO, IMAGES_PSEUDO, IMAGES_PSEUDO_SRC)

# copy_images(MASKS_HARD, IMAGES_HARD, IMAGES_HARD_SRC)
