
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrait des features 'haar' (ondelette discrete Haar) depuis les patches 64x128.
- Grille 8x4 (cellules 8x32 px) sur l'image
- Niveaux DWT: 3
- Sous-bandes utilisées: LH, HL, HH
- Statistiques par cellule: énergie (somme des carrés)
Sorties:
  X.npy (float32)  : matrice (N, D)
  y.npy (int8)     : 1 pour pos, 0 pour neg
  chemin.npy (obj) : chemins des images correspondantes
  meta.json        : infos sur les paramètres
"""
import os, argparse, glob, json
import numpy as np
from PIL import Image
import pywt

def wavelet_energy_grid(img_gray, levels=3, grid=(8,4), wavelet='haar'):
    H, W = img_gray.shape  # (128,64)
    gy, gx = grid  # 4 colonnes, 8 lignes? (défini comme (rows, cols))
    # Clarif: on définit grid=(8,4): 8 lignes, 4 colonnes
    gy, gx = grid
    cell_h = H // gy
    cell_w = W // gx
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=levels)
    # coeffs: [LL_L, (LH_L, HL_L, HH_L), ..., (LH_1, HL_1, HH_1)]
    feats = []
    for lev in range(1, levels+1):
        (LH, HL, HH) = coeffs[lev]
        for band in (LH, HL, HH):
            # redimensionner chaque sous-bande au HxW original par interpolation bilinéaire
            # pour un partitionnement en cellules cohérent
            bimg = Image.fromarray(np.float32(np.abs(band)))
            bimg = bimg.resize((W, H), Image.BILINEAR)
            b = np.asarray(bimg, dtype=np.float32)
            # énergie par cellule
            for iy in range(gy):
                for ix in range(gx):
                    y1 = iy*cell_h; y2 = (iy+1)*cell_h if iy<gy-1 else H
                    x1 = ix*cell_w; x2 = (ix+1)*cell_w if ix<gx-1 else W
                    patch = b[y1:y2, x1:x2]
                    e = np.sum(patch*patch, dtype=np.float64)
                    feats.append(e)
    return np.array(feats, dtype=np.float32)

def load_images(dirpath, label):
    paths = sorted(glob.glob(os.path.join(dirpath, "*.png")) + glob.glob(os.path.join(dirpath, "*.jpg")))
    X = []
    ys = []
    chemins = []
    for p in paths:
        try:
            img = Image.open(p).convert("L").resize((64,128), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            X.append(wavelet_energy_grid(arr, levels=3, grid=(8,4), wavelet='haar'))
            ys.append(label)
            chemins.append(p)
        except Exception:
            continue
    if not X:
        return np.empty((0,0), dtype=np.float32), np.array([], dtype=np.int8), []
    return np.vstack(X), np.array(ys, dtype=np.int8), chemins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="Dossier patches positifs")
    ap.add_argument("--neg", required=True, help="Dossier patches négatifs")
    ap.add_argument("--outdir", required=True, help="Dossier sortie pour X/y/chemin/meta")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    Xpos, ypos, cpos = load_images(args.pos, 1)
    Xneg, yneg, cneg = load_images(args.neg, 0)

    if Xpos.size == 0 and Xneg.size == 0:
        raise RuntimeError("Aucun patch trouvé dans les dossiers fournis.")

    if Xpos.size == 0:
        X = Xneg; y = yneg; chemins = cneg
    elif Xneg.size == 0:
        X = Xpos; y = ypos; chemins = cpos
    else:
        X = np.vstack([Xpos, Xneg])
        y = np.concatenate([ypos, yneg])
        chemins = cpos + cneg

    # Sauvegardes
    np.save(os.path.join(args.outdir, "X.npy"), X.astype(np.float32))
    np.save(os.path.join(args.outdir, "y.npy"), y.astype(np.int8))
    np.save(os.path.join(args.outdir, "chemin.npy"), np.array(chemins, dtype=object))

    meta = {
        "feature": "haar_wavelet",
        "levels": 3,
        "grid": [8,4],
        "bands": ["LH","HL","HH"],
        "window": [64,128],
        "pos_count": int((y==1).sum()),
        "neg_count": int((y==0).sum()),
        "dim": int(X.shape[1]),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK: X shape {X.shape}, pos {meta['pos_count']}, neg {meta['neg_count']}, dim={meta['dim']}")
    print(f"Sauvé dans {args.outdir}")

if __name__ == "__main__":
    main()
    
    
'''
python /home/ousman/links/scratch/pieton_tm/scripts/feature_haar_inria.py   --pos /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/positifs   --neg /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/negatifs   --outdir /home/ousman/links/scratch/pieton_tm/gabarit/features_haar_inria_train_64x128
'''