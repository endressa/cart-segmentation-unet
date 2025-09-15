# mine_hard_cases.py
import os, json, math, random
from pathlib import Path
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import pandas as pd
from tqdm import tqdm

# ---------------- CONFIG ----------------
CHECKPOINT = Path("/home/sarah/Documents/background_segmentation/cart_segmentation/cart-segmentation-unet/pseudo_model_12.pth")  # <- set me
IMAGES_ROOT = Path("/home/sarah/Documents/background_segmentation/whole_datasets/ariel_sarah/raw")    # unlabeled or pseudo-labeled pool
OUT_DIR = Path("/home/sarah/Documents/background_segmentation/whole_datasets/hard_mining_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model / data params (match training!)
IMG_SIZE = (512, 512)
SIDE_PADDING_RATIO = 0.1
THRESHOLD = 0.44      # use your current best_t; adjust after each stage
USE_IMAGENET_NORM = True   # True = match training; False = 0-1 only

# Mining params
DELTA_NEAR = 0.05     # |p-0.5| < DELTA counts as "ambiguous"
TTA_FLIPS = [False]  # run original + hflip
SEED = 42
SAVE_PREVIEWS = False
N_PREVIEWS = 100      # cap # of preview PNGs to save (top by composite score)
SAMPLE_SIZE = 5000   # set None to use all images


# ---------------- Letterbox like training ----------------
def letterbox_image_with_side_padding(image, padding_color=(0,0,0), side_padding_ratio=SIDE_PADDING_RATIO):
    h, w = image.shape[:2]
    side_padding = round(w * side_padding_ratio)
    padded_w = w + 2 * side_padding
    padded_h = h
    padded = np.full((padded_h, padded_w, 3), padding_color, dtype=np.uint8)
    padded[:, side_padding:side_padding+w] = image
    max_dim = max(padded_w, padded_h)
    canvas = np.full((max_dim, max_dim, 3), padding_color, dtype=np.uint8)
    xo = (max_dim - padded_w)//2
    yo = (max_dim - padded_h)//2
    canvas[yo:yo+padded_h, xo:xo+padded_w] = padded
    return canvas

def make_transform():
    if USE_IMAGENET_NORM:
        mean=(0.485, 0.456, 0.406); std=(0.229, 0.224, 0.225)
    else:
        mean=(0,0,0); std=(1,1,1)
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1], interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# ---------------- Model ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(ckpt_path):
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet",
                     in_channels=3, classes=1, activation=None).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# ---------------- Metrics ----------------
def mean_entropy(p, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return float((-(p*np.log(p) + (1-p)*np.log(1-p))).mean())

def frac_near_threshold(p, delta=0.05):
    return float((np.abs(p-0.5) < delta).mean())

def tta_variance_mean(probs_list):
    # probs_list: list of HxW arrays
    stack = np.stack(probs_list, axis=0)  # [T,H,W]
    return float(stack.var(axis=0).mean())

def binarize(p, t):
    return (p > t).astype(np.uint8)

def perimeter_over_area(mask01):
    area = int(mask01.sum())
    if area == 0: return 0.0
    cnts,_ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return float(per / (area + 1e-7))

def count_components(mask01):
    num, _, _, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), 8)
    return int(max(0, num-1))  # exclude background

def hole_mask(mask01):
    m = mask01.astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    H, W = m.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    holes = np.zeros_like(m)
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        touches = (x==0) or (y==0) or (x+w==W) or (y+h==H)
        if not touches:
            holes[labels == i] = 1
    return holes  # {0,1}

def hole_fraction(mask01):
    holes = hole_mask(mask01)
    a = mask01.sum()
    return 0.0 if a == 0 else float(holes.sum() / a)

# ---------------- Inference ----------------
@torch.no_grad()
def predict_probs(model, img_bgr, transform):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb  = letterbox_image_with_side_padding(img_rgb)
    aug = transform(image=img_lb)
    x = aug["image"].unsqueeze(0).to(DEVICE)
    logits = model(x)
    p = torch.sigmoid(logits)[0,0].detach().cpu().numpy().astype(np.float32)
    return p  # HxW in [0,1]

@torch.no_grad()
def predict_probs_tta(model, img_bgr, transform, flips=(False, True)):
    outs = []
    for f in flips:
        img = cv2.flip(img_bgr, 1) if f else img_bgr
        p = predict_probs(model, img, transform)
        if f:
            p = np.flip(p, axis=1)
        outs.append(p)
    return outs

# ---------------- Mining pipeline ----------------
def mine(IMAGES_ROOT, out_dir=OUT_DIR):
    tfm = make_transform()
    model = load_model(CHECKPOINT)

    # gather images recursively
    image_exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    imgs = [p for p in Path(IMAGES_ROOT).rglob("*") if p.suffix.lower() in image_exts]
    imgs.sort()
    print(f"Found {len(imgs)} images total.")

    # reproducible random subset
    rng = random.Random(SEED)
    if SAMPLE_SIZE is not None and len(imgs) > SAMPLE_SIZE:
        imgs = rng.sample(imgs, SAMPLE_SIZE)
        print(f"Sampling {len(imgs)} images (SAMPLE_SIZE={SAMPLE_SIZE}, seed={SEED}).")
        (out_dir / "sampling").mkdir(parents=True, exist_ok=True)
        sampled_txt = out_dir / "sampling" / f"sampled_images_seed{SEED}_n{len(imgs)}.txt"
        with sampled_txt.open("w") as f:
            for p in tqdm(imgs, desc="Saving sampled list", unit="img"):
                f.write(str(p) + "\n")
        print(f"Saved sampled paths to: {sampled_txt}")
    else:
        print("Using all images (no sampling).")

    rows = []
    preview_dir = out_dir / "previews"
    if SAVE_PREVIEWS:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # --- tqdm progress for scoring ---
    for ip in tqdm(imgs, desc="Scoring images", unit="img"):
        img_bgr = cv2.imread(str(ip))
        if img_bgr is None:
            continue

        # TTA probs
        probs_list = predict_probs_tta(model, img_bgr, tfm, flips=TTA_FLIPS)
        p_mean = np.mean(probs_list, axis=0)
        p_var_mean = tta_variance_mean(probs_list)

        # binarize
        m01 = binarize(p_mean, THRESHOLD)

        # metrics
        ent = mean_entropy(p_mean)
        near = frac_near_threshold(p_mean, delta=DELTA_NEAR)
        comp = count_components(m01)
        hole_frac = hole_fraction(m01)
        per_area = perimeter_over_area(m01)

        # composite score (higher = harder)
        score = (0.35*ent + 0.25*near + 0.20*p_var_mean + 0.10*hole_frac + 0.10*min(per_area/0.2, 1.0)) \
                + 0.05*min(comp, 3)

        rows.append({
            "path": str(ip),
            "entropy_mean": ent,
            "frac_near_0p5": near,
            "tta_var_mean": p_var_mean,
            "components": comp,
            "hole_fraction": hole_frac,
            "perimeter_over_area": per_area,
            "composite": score
        })

    # rank & save
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows to save. Check image paths.")
        return None

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    csv_path = out_dir / "hard_candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved ranking to: {csv_path}")

    # --- tqdm progress for previews (optional) ---
    if SAVE_PREVIEWS:
        top = df.head(N_PREVIEWS)
        for _, row in tqdm(top.iterrows(), total=len(top), desc="Saving previews", unit="img"):
            ip = Path(row["path"])
            img_bgr = cv2.imread(str(ip))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lb  = letterbox_image_with_side_padding(img_rgb)
            p = predict_probs(model, img_bgr, tfm)
            m = (p > THRESHOLD).astype(np.uint8)

            amb = (np.abs(p-0.5) < DELTA_NEAR).astype(np.uint8) * 255
            hm_color = cv2.applyColorMap(amb.astype(np.uint8), cv2.COLORMAP_JET)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            hm_color = cv2.addWeighted(img_lb, 0.65, hm_color, 0.35, 0)

            mask_rgb = img_lb.copy()
            m255 = (m*255).astype(np.uint8)
            cnts,_ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_rgb, cnts, -1, (255,0,0), 2)

            holes = hole_mask(m).astype(np.uint8)*255
            if holes.any():
                cnts,_ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_rgb, cnts, -1, (255,255,0), 2)

            vis = np.concatenate([
                cv2.resize(img_lb, (IMG_SIZE[1], IMG_SIZE[0])),
                cv2.resize(mask_rgb, (IMG_SIZE[1], IMG_SIZE[0])),
                cv2.resize(hm_color, (IMG_SIZE[1], IMG_SIZE[0]))
            ], axis=1)

            base = "__".join(ip.parts[-3:]).replace(os.sep, "__")
            out_png = preview_dir / f"{base}.png"
            cv2.imwrite(str(out_png), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        print(f"Saved {len(top)} preview PNGs to: {preview_dir}")

    return df



if __name__ == "__main__":
    mine(IMAGES_ROOT, OUT_DIR)
