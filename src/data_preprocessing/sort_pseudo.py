import shutil
from pathlib import Path

# Quelle und Ziel
SRC = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_finetuned/head_and_shoulders_sub_sarah/raw")
DST = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_2")

# alle .png-Dateien durchsuchen, die keine Overlays oder Probs sind
png_files = list(SRC.rglob("*.png"))
mask_files = [f for f in png_files if not (f.name.endswith("_overlay_512.png") or f.name.endswith("_probs_512.png"))]

print(f"Found {len(mask_files)} mask files to copy.")

for src_file in mask_files:
    rel_path = src_file.relative_to(SRC)   # relative Pfad von SRC
    dst_file = DST / rel_path              # Zielpfad unter DST
    dst_file.parent.mkdir(parents=True, exist_ok=True)  # Ordner anlegen
    shutil.copy2(src_file, dst_file)       # Datei kopieren (mit Metadata)

print(f"✅ Done. Copied {len(mask_files)} masks to {DST}")
