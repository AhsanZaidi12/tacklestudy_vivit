"""
LEGACY PRELIMINARY-DATA UTILITY

This script was developed for the earlier approximately 190-clip dataset.
It creates one fixed train/validation split, generates 86 additional training
clips, and prepares 19 configurations.

It was not used to construct the final 733-clip, stratified five-fold
experiment reported in the ICPR 2026 paper. The final five-fold preparation
pipeline and exact fold manifests are maintained separately and are being
prepared for public release.

Do not use this script to reproduce the paper's final headline results.
It is retained only for provenance and historical reference.
"""





#!/usr/bin/env python3
"""
build_taguchi_datasets.py  ·  v3
--------------------------------
python Taguchi_datasets.py \
    --video_root "/mnt/e/Tackle_Ablation/videos" \
    --label_csv  "/mnt/e/Tackle_Ablation/label.csv" \
    --output_root "/mnt/e/Tackle_Ablation/taguchi_runs"

"""

import argparse, os, shutil, random, yaml
from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm

# ────────────── video helpers (unchanged) ──────────────
def read_video(path: Path) -> np.ndarray:
    cap, frames = cv2.VideoCapture(str(path)), []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.asarray(frames)

def save_video(frames: np.ndarray, out_path: Path, fps=30) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()

# ────────────── augmentation kernels (unchanged) ──────────────
def add_noise(frames, std=0.2):
    noise = np.random.normal(0, std * 255, frames.shape)
    return np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def adjust_brightness(frames, factor):
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return np.asarray(out)

def rotate_video(frames, direction):
    angle = 10 if direction == "L" else -10
    h, w = frames[0].shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return np.asarray([cv2.warpAffine(f, M, (w, h)) for f in frames])

def flip_video(frames, direction):
    code = 1 if direction == "H" else 0
    return np.asarray([cv2.flip(f, code) for f in frames])

def apply_taguchi(frames, cfg):
    if cfg["brightness"] == "inc":
        frames = adjust_brightness(frames, 1.3)
    elif cfg["brightness"] == "dec":
        frames = adjust_brightness(frames, 0.7)
    if cfg["rotate"] in {"L", "R"}:
        frames = rotate_video(frames, cfg["rotate"])
    if cfg["flip"]   in {"H", "V"}:
        frames = flip_video(frames, cfg["flip"])
    if cfg["noise"]:
        frames = add_noise(frames)
    return frames

# ────────────── Taguchi array + baseline ──────────────
L18 = [
    [0,1,1,1],[0,1,2,2],[0,1,3,3],[0,2,1,2],[0,2,2,3],[0,2,3,1],
    [0,3,1,3],[0,3,2,1],[0,3,3,2],[1,1,1,3],[1,1,2,1],[1,1,3,2],
    [1,2,1,1],[1,2,2,2],[1,2,3,3],[1,3,1,2],[1,3,2,3],[1,3,3,1]
]
ALL_RUNS = [[0,3,3,3]] + L18          # prepend baseline

def decode(A,B,C,D):
    return dict(
        noise = bool(A),
        brightness = {1:"inc", 2:"dec", 3:None}[B],
        rotate = {1:"L", 2:"R", 3:None}[C],
        flip   = {1:"H", 2:"V", 3:None}[D]
    )

# ────────────── main workflow ──────────────
def main(args):
    rng = random.Random(args.seed)

    # 1) read label CSV and auto-detect headers ------------------------------
    df = pd.read_csv(args.label_csv)
    file_col  = args.file_col  or next(c for c in df.columns if c.lower() in {"fname","file","video","video_path"})
    label_col = args.label_col or next(c for c in df.columns if c.lower() in {"label","class","target"})
    df = df[[file_col, label_col]].rename(columns={file_col:"fname", label_col:"label"})
    df["label"] = df["label"].astype(int)

    # 2) reproducible 80-20 split with 20+20 validation ----------------------
    safe, risky = df[df.label==0], df[df.label==1]
    val_df = pd.concat([safe.sample(20, random_state=args.seed),
                        risky.sample(20, random_state=args.seed)])
    train_df = df.drop(val_df.index).reset_index(drop=True)   # 113 S + 37 R

    # 3) fixed augmentation manifest (10 safe + 76 risky) --------------------
    risky_train = train_df[train_df.label==1].copy()
    risky_train["copies"] = 2
    extra2 = rng.sample(list(risky_train.index), k=2)
    risky_train.loc[extra2, "copies"] += 1
    safe_aug = train_df[train_df.label==0].sample(10, random_state=args.seed).copy()
    safe_aug["copies"] = 1
    aug_manifest = pd.concat([risky_train, safe_aug]).reset_index(drop=True)

    # 4) run-catalog for Excel ----------------------------------------------
    catalog_rows = []

    # 5) iterate over baseline + 18 Taguchi rows ----------------------------
    for ridx, row in enumerate(ALL_RUNS):
        run_id = f"run_{ridx:02d}"
        run_dir = Path(args.output_root) / run_id
        train_vid_dir = run_dir / "train/videos"
        val_vid_dir   = run_dir / "val/videos"

        catalog_rows.append({"run_id": run_id, "A": row[0], "B": row[1], "C": row[2], "D": row[3]})

        # skip if already processed
        if (run_dir / "train_labels.csv").exists():
            print(f"[{run_id}] exists – skipping")
            continue

        cfg = decode(*row)
        train_vid_dir.mkdir(parents=True, exist_ok=True)
        val_vid_dir.mkdir(parents=True, exist_ok=True)

        # 5a) copy originals
        for _, r in train_df.iterrows():
            shutil.copy2(Path(args.video_root)/r.fname, train_vid_dir/r.fname)
        for _, r in val_df.iterrows():
            shutil.copy2(Path(args.video_root)/r.fname, val_vid_dir/r.fname)

        # 5b) generate augmented clips (86)
        aug_entries = []
        for _, r in tqdm(aug_manifest.iterrows(), total=len(aug_manifest), desc=f"[{run_id}] augment"):
            src = Path(args.video_root)/r.fname
            frames = read_video(src)
            if len(frames) < 32:
                frames = np.concatenate([frames] + [frames[-1:]]*(32-len(frames)))
            frames = frames[:32]
            for cidx in range(r.copies):
                new_name = f"{Path(r.fname).stem}_{run_id}_{cidx:02d}.mp4"
                out_path = train_vid_dir / new_name
                save_video(apply_taguchi(frames.copy(), cfg), out_path)
                aug_entries.append({"fname": new_name, "label": int(r.label)})

        # 5c) save manifests & recipe
        pd.concat([train_df, pd.DataFrame(aug_entries)]).to_csv(run_dir/"train_labels.csv", index=False)
        val_df.to_csv(run_dir/"val_labels.csv", index=False)
        with open(run_dir/"augment_recipe.yml","w") as f:
            yaml.safe_dump(cfg, f)
        print(f"[{run_id}] ✔ created")

    # 6) write Excel catalogue ----------------------------------------------
    catalog_path = Path(args.output_root)/"run_catalog.xlsx"
    pd.DataFrame(catalog_rows).to_excel(catalog_path, index=False)
    print(f"Run catalogue saved → {catalog_path}")

# ────────────── CLI ──────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root",  required=True)
    ap.add_argument("--label_csv",   required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--file_col",  type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
