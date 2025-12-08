#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features_hog_inria.py
Calcule descripteurs HOG (64x128, cell=8, block=2x2, 9 bins) sur des patches positifs/négatifs
et sauvegarde X.npy, y.npy, chemins.npy, meta.json.

Exemple (adapter les chemins) :
python features_hog_inria.py \  --pos /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/positifs \  --neg /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/negatifs \  --out_dir /home/ousman/links/scratch/pieton_tm/gabarit/features_hog_inria_64x128 \  --resize
"""
import argparse, os, json, numpy as np, cv2
from glob import glob

def imread_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Impossible de lire {path}")
    return im

def hog_64x128():
    return cv2.HOGDescriptor(
        _winSize=(64,128),
        _blockSize=(16,16),
        _blockStride=(8,8),
        _cellSize=(8,8),
        _nbins=9
    )

def list_images(dirpath):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = sum([glob(os.path.join(dirpath, e)) for e in exts], [])
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="dossier patches positifs")
    ap.add_argument("--neg", required=True, help="dossier patches négatifs")
    ap.add_argument("--out_dir", required=True, help="dossier sortie")
    ap.add_argument("--resize", action="store_true", help="forcer resize 64x128")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hog = hog_64x128()

    X, y, chemins = [], [], []

    for f in list_images(args.pos):
        im = imread_gray(f)
        if args.resize: im = cv2.resize(im, (64,128), interpolation=cv2.INTER_LINEAR)
        feat = hog.compute(im).reshape(-1)
        X.append(feat); y.append(1); chemins.append(f)

    for f in list_images(args.neg):
        im = imread_gray(f)
        if args.resize: im = cv2.resize(im, (64,128), interpolation=cv2.INTER_LINEAR)
        feat = hog.compute(im).reshape(-1)
        X.append(feat); y.append(0); chemins.append(f)

    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)
    chemins = np.asarray(chemins)

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    np.save(os.path.join(args.out_dir, "chemins.npy"), chemins)

    meta = {
        "feature": "HOG",
        "win": "64x128",
        "cell": "8x8",
        "block": "2x2",
        "bins": 9,
        "dim": int(X.shape[1]),
        "n_pos": int((y==1).sum()),
        "n_neg": int((y==0).sum())
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK: X shape {X.shape}, pos {meta['n_pos']}, neg {meta['n_neg']}, dim={meta['dim']}")
    print(f"Sauvé dans {args.out_dir}")

if __name__ == "__main__":
    main()

# ----------------------
# Commandes exemples :
# python features_hog_inria.py \#   --pos /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/positifs \#   --neg /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/negatifs \#   --out_dir /home/ousman/links/scratch/pieton_tm/gabarit/features_hog_inria_64x128 \#   --resize
# ----------------------
