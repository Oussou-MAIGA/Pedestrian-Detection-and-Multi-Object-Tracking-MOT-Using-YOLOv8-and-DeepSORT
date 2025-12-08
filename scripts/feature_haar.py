#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features_haar.py
- Parcourt les patchs positifs/négatifs 64x128 (ou redimensionne au besoin)
- Calcule ondelettes de Haar (DWT2, on ignore LL, on concatène LH+HL+HH)
- Normalise: z-score de l'image avant DWT + L2 du vecteur après DWT
- Sauvegarde X.npy, y.npy, chemins.npy, meta.json
"""

import argparse, json
from pathlib import Path
import numpy as np
import cv2, pywt

def lister_images(dossier: Path):
    exts = (".png",".jpg",".jpeg",".PNG",".JPG",".JPEG")
    return sorted([p for p in dossier.rglob("*") if p.suffix in exts], key=lambda p: [int(t) if t.isdigit() else t for t in __import__('re').split(r'(\d+)', p.name)])

def zscore(img):
    x = img.astype(np.float32)
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-6)

def desc_haar(image, ondelette="haar", cible=(64,128)):
    # image: np.uint8/float32, niveaux de gris
    if image.shape[:2] != (cible[1], cible[0]):
        image = cv2.resize(image, cible, interpolation=cv2.INTER_LINEAR)
    X = zscore(image)
    LL, (LH, HL, HH) = pywt.dwt2(X, ondelette)
    feat = np.concatenate([LH.ravel(), HL.ravel(), HH.ravel()]).astype(np.float32)
    n = np.linalg.norm(feat) + 1e-8
    return feat / n  # L2-normalisé

def main():
    ap = argparse.ArgumentParser(description="Extraction features Haar (LH,HL,HH)")
    ap.add_argument("--racine", type=Path, required=True,
                    help="racine dataset_caltech/train contenant positives/ et negatives/")
    ap.add_argument("--taille", type=str, default="64x128",
                    help="taille patch LxH, défaut 64x128")
    ap.add_argument("--ondelette", type=str, default="haar")
    ap.add_argument("--sorties", type=Path, required=True,
                    help="dossier de sortie pour X.npy, y.npy, chemins.npy, meta.json")
    args = ap.parse_args()

    L,H = [int(t) for t in args.taille.lower().split("x")]
    pos_dir = args.racine / "positifs_qualite"
    neg_dir = args.racine / "negatifs_qualite"
    if not pos_dir.is_dir() or not neg_dir.is_dir():
        raise SystemExit(f"Dossiers introuvables: {pos_dir} / {neg_dir}")

    chemins_pos = lister_images(pos_dir)
    chemins_neg = lister_images(neg_dir)

    X, y, chemins = [], [], []
    # positifs = +1
    for p in chemins_pos:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        f = desc_haar(im, args.ondelette, (L,H))
        X.append(f); y.append(1); chemins.append(str(p))
    # négatifs = -1
    for p in chemins_neg:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        f = desc_haar(im, args.ondelette, (L,H))
        X.append(f); y.append(-1); chemins.append(str(p))

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int8)
    chemins = np.array(chemins)

    args.sorties.mkdir(parents=True, exist_ok=True)
    np.save(args.sorties/"X.npy", X)
    np.save(args.sorties/"y.npy", y)
    np.save(args.sorties/"chemins.npy", chemins)

    meta = {
        "taille_patch": [L,H],
        "ondelette": args.ondelette,
        "nb_pos": int((y==1).sum()),
        "nb_neg": int((y==-1).sum()),
        "dim_vecteur": int(X.shape[1]),  # attend ~3*(L/2)*(H/2) = 3*(L*H/4)
    }
    (args.sorties/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK features →", args.sorties)
    print(meta)

if __name__ == "__main__":
    main()
    
# python feature_haar.py --racine ~/links/scratch/pieton_tm/dataset_caltech/train/ --sorties ~/links/scratch/pieton_tm/gabarit/features_haar_64x128/