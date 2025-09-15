#!/usr/bin/env python3
# Hard-coded version: ranks masks by soft-probability metrics and copies top 1300
# while preserving the original "store/session/images" structure.
#
# Edit the constants below if your paths or parameters change.

from pathlib import Path
import numpy as np
import csv
import shutil
import sys

# ------------------- HARD-CODED SETTINGS -------------------
SRC_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_15/head_and_shoulders_sub_sarah/raw")
DST_ROOT = Path("/home/sarah/Documents/background_segmentation/dataset/pseudo_masks_15_best")
TOP_N    = 1300
THR      = 0.5
# -----------------------------------------------------------

def entropy_map(p, eps=1e-6):
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

def compute_metrics(prob, threshold):
    fg_area = float((prob >= threshold).mean())
    fg_conf = float(np.median(np.abs(prob - 0.5) * 2.0))   # 0..1 (higher is better)
    mean_ent = float(entropy_map(prob).mean())             # 0..~0.693 (lower is better)
    return fg_area, fg_conf, mean_ent

def score_sample(fg_area, fg_conf, mean_ent):
    # Soft constraints similar to your QC:
    MIN_FG_AREA, MAX_FG_AREA = 0.005, 0.60
    area_penalty = 0.0
    if fg_area < MIN_FG_AREA:
        area_penalty = -min(1.0, (MIN_FG_AREA - fg_area) / MIN_FG_AREA)
    elif fg_area > MAX_FG_AREA:
        area_penalty = -min(1.0, (fg_area - MAX_FG_AREA) / (1.0 - MAX_FG_AREA))
    return float(0.7 * fg_conf - 0.5 * mean_ent + 0.3 * area_penalty)

def find_pairs(src_root: Path):
    for prob_path in src_root.rglob("*_prob.npz"):
        stem = prob_path.stem
        if not stem.endswith("_prob"):
            continue
        mask_stem = stem[:-5]   # remove "_prob"
        mask_path = prob_path.with_name(mask_stem + ".png")
        if not mask_path.exists():
            # no corresponding mask -> skip
            continue
        rel_dir = mask_path.parent.relative_to(src_root)
        yield prob_path, mask_path, rel_dir

def main():
    if not SRC_ROOT.exists():
        print(f"[ERROR] Source path does not exist: {SRC_ROOT}", file=sys.stderr)
        return 2

    pairs = list(find_pairs(SRC_ROOT))
    if not pairs:
        print(f"[WARN] No *_prob.npz + .png pairs found under: {SRC_ROOT}", file=sys.stderr)
        return 1

    print(f"Found {len(pairs)} mask/prob pairs. Computing metrics...")

    rows = []
    bad = 0
    for prob_path, mask_path, rel_dir in pairs:
        try:
            with np.load(prob_path) as npz:
                prob = npz["prob"] if "prob" in npz.files else None
            if prob is None:
                bad += 1
                continue
        except Exception as e:
            print(f"[WARN] Failed to load {prob_path}: {e}", file=sys.stderr)
            bad += 1
            continue

        fg_area, fg_conf, mean_ent = compute_metrics(prob, THR)
        score = score_sample(fg_area, fg_conf, mean_ent)
        rows.append((prob_path, mask_path, rel_dir, fg_area, fg_conf, mean_ent, score))

    if not rows:
        print("[WARN] No valid entries after metric computation.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r[-1], reverse=True)
    top_rows = rows[:TOP_N]

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = DST_ROOT / "top_masks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank","score","fg_conf","mean_entropy","fg_area","mask_rel","prob_rel"])
        for rank, (prob_path, mask_path, rel_dir, fg_area, fg_conf, mean_ent, score) in enumerate(top_rows, 1):
            mask_rel = str(mask_path.relative_to(SRC_ROOT))
            prob_rel = str(prob_path.relative_to(SRC_ROOT))
            w.writerow([rank, f"{score:.6f}", f"{fg_conf:.6f}", f"{mean_ent:.6f}", f"{fg_area:.6f}", mask_rel, prob_rel])

    copied = 0
    for prob_path, mask_path, rel_dir, *_ in top_rows:
        dest_dir = DST_ROOT / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(mask_path, dest_dir / mask_path.name)
        except Exception as e:
            print(f"[WARN] Copy mask failed {mask_path} -> {dest_dir / mask_path.name}: {e}", file=sys.stderr)
        try:
            shutil.copy2(prob_path, dest_dir / prob_path.name)
        except Exception as e:
            print(f"[WARN] Copy prob failed {prob_path} -> {dest_dir / prob_path.name}: {e}", file=sys.stderr)
        copied += 1

    print(f"Wrote ranking CSV: {csv_path}")
    print(f"Copied top {copied} masks (and their npz) into: {DST_ROOT}")
    if bad:
        print(f"Skipped {bad} broken/invalid npz files.")
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())