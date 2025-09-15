# cut_mask_padding.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Same parameters as your mask generation
SIDE_PADDING_RATIO = 0.1

    # Paths
# IMAGES_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/images_pseudo")

# IMAGE_ROOTS = [
#     Path("/opt/whizcart/shared/carrefour_classes/images/merci_raw"),
#     Path("/opt/whizcart/shared/carrefour_classes/images/gemuese_netz/raw"),
#     Path("/opt/whizcart/shared/carrefour_classes/images/head_and_shoulders_sub_sarah/raw"),
#     Path("/opt/whizcart/shared/carrefour_classes/images/ariel_sarah/ariel_sarah/raw"),
# ]

# MASKS_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/mixed_pseudo_clean")
# OUTPUT_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/mixed_pseudo_clean_unlettered")

# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def calculate_content_boundaries(orig_w, orig_h, final_w, final_h, side_padding_ratio=SIDE_PADDING_RATIO):
    # 1) Seitpadding
    side_padding = round(orig_w * side_padding_ratio)
    padded_w = orig_w + 2 * side_padding
    padded_h = orig_h

    # 2) Quadrat + Offsets
    max_dim = max(padded_w, padded_h)
    x_offset = (max_dim - padded_w) // 2
    y_offset = (max_dim - padded_h) // 2

    # 3) Content-Bereich im Quadrat (vor Resize)
    x0 = x_offset + side_padding
    y0 = y_offset
    x1 = x0 + orig_w
    y1 = y0 + orig_h

    # 4) Skalierung auf finale Maske (kann rechteckig sein – der Vollständigkeit halber)
    sx = final_w / max_dim
    sy = final_h / max_dim
    X0 = int(round(x0 * sx)); Y0 = int(round(y0 * sy))
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))

    # Clamp
    X0 = max(0, min(final_w, X0)); X1 = max(0, min(final_w, X1))
    Y0 = max(0, min(final_h, Y0)); Y1 = max(0, min(final_h, Y1))
    return X0, Y0, X1, Y1

def cut_mask_padding(mask, orig_w, orig_h, side_padding_ratio=SIDE_PADDING_RATIO):
    """
    Remove padding completely and return a mask cropped
    to the original image size (orig_h × orig_w).
    """
    H, W = mask.shape[:2]
    x0, y0, x1, y1 = calculate_content_boundaries(orig_w, orig_h, W, H, side_padding_ratio)

    # Crop to the content region
    cropped = mask[y0:y1, x0:x1]

    # Resize to exactly match original dimensions (safety, in case of rounding)
    cropped_resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return cropped_resized


def get_original_dimensions_cv2(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    h, w = img.shape[:2]   # OpenCV: (H,W,3)
    return w, h


def get_original_dimensions(image_path):
    """Get original image dimensions."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"⚠️ Could not read {image_path}: {e}")
        return None, None

def fill_small_holes(fg_mask_255, max_hole_area_frac=0.004):
    """
    Fills background components fully enclosed by the foreground (holes).
    Only fills holes up to a size threshold set as a fraction of image area.
    fg_mask_255: uint8 {0,255}
    """
    m = (fg_mask_255 > 127).astype(np.uint8) * 255
    H, W = m.shape
    max_area = int(H * W * max_hole_area_frac)  # ~0.4% by default

    # connected components on the *background*
    inv = (255 - m)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    filled = m.copy()
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        # if the background component touches the image border, it's not a hole
        touches_border = (x == 0) or (y == 0) or (x + w == W) or (y + h == H)
        if touches_border:
            continue
        if area <= max_area:
            filled[labels == i] = 255  # fill the hole

    return filled

def post_clean(binary_mask, min_area=1500, open_ks=5, close_ks=9,
               hole_frac=0.004):
    m = (binary_mask > 127).astype(np.uint8) * 255

    # 1) remove thin spurs
    if open_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    # 2) keep only largest component (since you expect a single object)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num > 1:
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        keep = max(areas, key=lambda x: x[1])[0]
        m = np.where(labels == keep, 255, 0).astype(np.uint8)

    # 3) close small gaps on the main blob
    if close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # 4) fill residual holes inside the blob
    m = fill_small_holes(m, max_hole_area_frac=hole_frac)
    return m




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

def visualize_before_after(img_path, mask_path, cleaned_mask_path):
    """Visualize original vs cleaned mask for verification."""
    import matplotlib.pyplot as plt
    
    # Load data
    img = cv2.imread(str(img_path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask_orig = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask_clean = cv2.imread(str(cleaned_mask_path), cv2.IMREAD_GRAYSCALE)
    
    if any(x is None for x in [img, mask_orig, mask_clean]):
        print("⚠️ Could not load files for visualization")
        return
    
    # Get original dimensions and apply same letterboxing as training
    orig_width, orig_height = img.shape[1], img.shape[0]
    
    # Apply letterboxing to image (same as in your pipeline)
    img_lb = letterbox_image_with_side_padding(img, padding_color=(0,0,0))
    img_resized = cv2.resize(img_lb, (224, 224))
    
    # Resize masks to match
    mask_orig_resized = cv2.resize(mask_orig, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask_clean_resized = cv2.resize(mask_clean, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Create overlays
    overlay_orig = img_resized.copy()
    overlay_clean = img_resized.copy()
    
    overlay_orig[mask_orig_resized > 127] = [255, 0, 0]  # Red for original
    overlay_clean[mask_clean_resized > 127] = [0, 255, 0]  # Green for cleaned
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0,0].imshow(img_resized)
    axes[0,0].set_title("Letterboxed Image")
    
    axes[0,1].imshow(mask_orig_resized, cmap='gray')
    axes[0,1].set_title("Original Mask")
    
    axes[0,2].imshow(overlay_orig)
    axes[0,2].set_title("Original Overlay")
    
    axes[1,0].imshow(img_resized)
    axes[1,0].set_title("Same Image")
    
    axes[1,1].imshow(mask_clean_resized, cmap='gray')
    axes[1,1].set_title("Cleaned Mask")
    
    axes[1,2].imshow(overlay_clean)
    axes[1,2].set_title("Cleaned Overlay")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"{img_path.name}", y=0.98)
    plt.show()


# --- NEU: Utils für Overlay + Auto-Visualisierung ---
import random
import matplotlib.pyplot as plt

def make_overlay(img_rgb, mask_gray, color=(255, 0, 0), alpha=0.35, outline=True):
    """Legt Maske als halbtransparente Farbe + optional Kontur aufs Bild."""
    base = img_rgb.copy()
    m = (mask_gray > 127).astype(np.uint8)

    # Füllung
    color_arr = np.zeros_like(base)
    color_arr[..., 0] = color[0]; color_arr[..., 1] = color[1]; color_arr[..., 2] = color[2]
    base = (base * (1 - alpha) + color_arr * alpha * m[..., None]).astype(np.uint8)

    # Kontur
    if outline:
        cnts, _ = cv2.findContours((m*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base = cv2.drawContours(base, cnts, -1, color, thickness=3)
    return base

def visualize_examples(pairs, n=6, seed=42, resize=256, title="Preview cleaned masks"):
    """
    pairs: Liste von (img_path, orig_mask_path, cleaned_mask_path)
    """
    if not pairs:
        print("⚠️ Keine Paare zum Visualisieren gefunden.")
        return

    rng = random.Random(seed)
    samp = pairs if len(pairs) <= n else rng.sample(pairs, n)

    cols = 3  # Bild | Original-Overlay | Cleaned-Overlay
    rows = len(samp)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*3.6))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, (img_p, orig_m_p, clean_m_p) in enumerate(samp):
        # Bild + Letterbox wie im Training
        img_bgr = cv2.imread(str(img_p));  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb  = letterbox_image_with_side_padding(img_rgb, padding_color=(0,0,0))
        img_res = cv2.resize(img_lb, (resize, resize), interpolation=cv2.INTER_LINEAR)

        # Masken (auf gleiche Größe)
        m_orig  = cv2.imread(str(orig_m_p),  cv2.IMREAD_GRAYSCALE)
        m_clean = cv2.imread(str(clean_m_p), cv2.IMREAD_GRAYSCALE)
        if m_orig is None or m_clean is None:
            for c in range(cols): axes[r, c].axis("off")
            continue
        m_orig_r  = cv2.resize(m_orig,  (resize, resize), interpolation=cv2.INTER_NEAREST)
        m_clean_r = cv2.resize(m_clean, (resize, resize), interpolation=cv2.INTER_NEAREST)

        # Overlays
        over_orig  = make_overlay(img_res, m_orig_r,  color=(255,0,0),   alpha=0.35)
        over_clean = make_overlay(img_res, m_clean_r, color=(0,255,0),   alpha=0.35)

        axes[r,0].imshow(img_res);    axes[r,0].set_title(img_p.name);      axes[r,0].axis("off")
        axes[r,1].imshow(over_orig);  axes[r,1].set_title("Original");       axes[r,1].axis("off")
        axes[r,2].imshow(over_clean); axes[r,2].set_title("Cleaned");        axes[r,2].axis("off")

    plt.suptitle(title, y=0.995, fontsize=12)
    plt.tight_layout()
    # optional: speichern neben OUTPUT_ROOT
    out_png = Path(OUTPUT_ROOT) / "_preview_cleaned_examples.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"📸 Preview gespeichert: {out_png}")
    plt.show()

def visualize_examples_to_files(
    pairs,
    out_dir,
    n=None,                # None = all pairs; or set e.g. 100
    seed=42,
    resize=256,
    dpi=180,
    depth=1,               # how many parent dirs to include in filename to avoid collisions
    alpha=0.35,
):
    """
    Save one 3-column figure per (img, orig_mask, cleaned_mask) pair.

    pairs: list of (img_path, orig_mask_path, cleaned_mask_path)
    out_dir: folder to save PNGs into (will be created)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs:
        print("⚠️ No pairs to visualize.")
        return []

    rng = random.Random(seed)
    sample = pairs if n is None or len(pairs) <= n else rng.sample(pairs, n)

    saved = []
    for (img_p, orig_m_p, clean_m_p) in sample:
        # Read inputs
        img_bgr = cv2.imread(str(img_p))
        m_orig  = cv2.imread(str(orig_m_p),  cv2.IMREAD_GRAYSCALE)
        m_clean = cv2.imread(str(clean_m_p), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or m_orig is None or m_clean is None:
            # skip silently but you can print a warning if you want
            continue

        # Letterbox (same as training) + resize for display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb  = letterbox_image_with_side_padding(img_rgb, padding_color=(0,0,0))
        img_res = cv2.resize(img_lb, (resize, resize), interpolation=cv2.INTER_LINEAR)

        # Resize masks to match
        m_orig_r  = cv2.resize(m_orig,  (resize, resize), interpolation=cv2.INTER_NEAREST)
        m_clean_r = cv2.resize(m_clean, (resize, resize), interpolation=cv2.INTER_NEAREST)

        # Overlays
        over_orig  = make_overlay(img_res, m_orig_r,  color=(255, 0, 0), alpha=alpha)
        over_clean = make_overlay(img_res, m_clean_r, color=(0, 255, 0), alpha=alpha)

        # Figure (1 row x 3 cols)
        fig, axes = plt.subplots(1, 3, figsize=(3*4.2, 1*3.6))
        axes[0].imshow(img_res);    axes[0].set_title(img_p.name); axes[0].axis("off")
        axes[1].imshow(over_orig);  axes[1].set_title("Original"); axes[1].axis("off")
        axes[2].imshow(over_clean); axes[2].set_title("Cleaned");  axes[2].axis("off")
        plt.tight_layout()

        # Build a safe filename that includes a few parent folders to avoid collisions
        parent_bits = "__".join(img_p.parent.parts[-depth:]) if depth > 0 else ""
        base = f"{parent_bits}__{img_p.stem}" if parent_bits else img_p.stem
        out_png = out_dir / f"{base}.png"

        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        saved.append(out_png)

    print(f"📸 Saved {len(saved)} previews to: {out_dir}")
    return saved

def collect_pairs_from_roots(
    img_root,
    MASKS_ROOT,
    OUTPUT_ROOT,
    image_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    mask_exts=(".png", ".jpg", ".jpeg"),
    require_cleaned=True,
):
    pairs = []

    def first_existing(base_path, exts):
        for ext in exts:
            p = base_path.with_suffix(ext)
            if p.exists():
                return p
        return None

    img_paths = [p for p in img_root.rglob("*") if p.suffix.lower() in image_exts]

    for img_file in img_paths:
        rel = img_file.relative_to(img_root)
        orig_mask    = first_existing(MASKS_ROOT / rel, mask_exts)
        cleaned_mask = first_existing(OUTPUT_ROOT / rel, mask_exts)

        if orig_mask is None:
            continue
        if require_cleaned and cleaned_mask is None:
            continue

        pairs.append((img_file, orig_mask, cleaned_mask, img_root))

    return pairs


if __name__ == "__main__":
    # IMAGE_ROOTS = [
    #     Path("/opt/whizcart/shared/carrefour_classes/images/merci_raw"),
    #     Path("/opt/whizcart/shared/carrefour_classes/images/gemuese_netz/raw"),
    #     Path("/opt/whizcart/shared/carrefour_classes/images/head_and_shoulders_sub_sarah/raw"),
    #     Path("/opt/whizcart/shared/carrefour_classes/images/ariel_sarah/ariel_sarah/raw"),
    # ]
    IMAGE_ROOTS = [Path("/opt/whizcart/shared/carrefour_classes/images")]


    # IMAGE_ROOTS = [Path("/home/sarah/Documents/background_segmentation/dataset/images_hard")
                #    ]
    MASKS_ROOT = Path("~/sarah/background_segmentation/dataset/mixed_pseudo_clean").expanduser()

    # MASKS_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/masks_hard")
    OUTPUT_ROOT = Path("~/sarah/background_segmentation/dataset/mixed_pseudo_clean_unlettered").expanduser()

    all_pairs = []
    for img_root in IMAGE_ROOTS:
        pairs = collect_pairs_from_roots(img_root, MASKS_ROOT, OUTPUT_ROOT, require_cleaned=False)
        all_pairs.extend(pairs)

    count = 0
    for (img_file, orig_mask, _, img_root) in tqdm(all_pairs, desc="Cutting padding"):
        try:
            ow, oh = get_original_dimensions_cv2(img_file)
            mask = cv2.imread(str(orig_mask), cv2.IMREAD_GRAYSCALE)
            cut = cut_mask_padding(mask, ow, oh, SIDE_PADDING_RATIO)

            rel = img_file.relative_to(img_root)
            out_path = (OUTPUT_ROOT / rel).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_path), cut)
            count += 1
        except Exception as e:
            print(f"⚠️ Error processing {img_file}: {e}")

    print(f"✅ Done. Wrote {count} cleaned masks into {OUTPUT_ROOT}")
