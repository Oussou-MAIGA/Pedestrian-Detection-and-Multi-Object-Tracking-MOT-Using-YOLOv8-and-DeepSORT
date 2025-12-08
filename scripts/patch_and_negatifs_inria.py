
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère des patches positifs (64x128) et négatifs à partir d'images + labels YOLO.
- Images: JPG/PNG etc. (supposées 640x640 ici, mais géré génériquement)
- Labels: fichiers .txt au format "class cx cy w h" avec valeurs relatives [0,1]
Sorties:
  patches/pos/*.png   (redimensionnés en 64x128)
  patches/neg/*.png   (fenêtres aléatoires 64x128 qui évitent les GT, IoU<0.2)
Args principaux:
  --images <dir>   répertoire des images
  --labels <dir>   répertoire des labels .txt
  --out_pos <dir>  répertoire de sortie pour les positifs
  --out_neg <dir>  répertoire de sortie pour les négatifs
  --neg_ratio 3    nombre de négatifs par positif (par défaut 3)
  --seed 123
Notes:
 - On ajuste chaque bbox vers le ratio 1:2 (64x128) par padding symétrique.
 - On recadre dans l'image, puis on redimensionne à 64x128 (bilinéaire).
 - On accepte plusieurs classes dans les .txt mais on ne garde que "piéton" si --classes est défini.
"""
import os, argparse, glob, math, random
from typing import List, Tuple
import numpy as np
from PIL import Image

def yolo_to_xyxy(rel, W, H):
    cx, cy, w, h = rel
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [x1, y1, x2, y2]

def clamp_box(box, W, H):
    x1,y1,x2,y2 = box
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return [x1,y1,x2,y2]

def pad_to_ratio(box, target_ratio=0.5, W=None, H=None):
    """
    Ajuste la boîte pour obtenir (w/h) ≈ target_ratio (ici 64/128 = 0.5).
    On pad symétriquement en gardant le centre.
    """
    x1,y1,x2,y2 = box
    w = x2-x1
    h = y2-y1
    if h <= 0 or w <= 0:
        return box
    r = w/float(h)
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    if r < target_ratio:
        # pas assez large → élargir w
        new_w = target_ratio * h
        dw = (new_w - w)/2.0
        x1 = cx - (w/2.0) - dw
        x2 = cx + (w/2.0) + dw
    elif r > target_ratio:
        # trop large → augmenter h
        new_h = (w/target_ratio)
        dh = (new_h - h)/2.0
        y1 = cy - (h/2.0) - dh
        y2 = cy + (h/2.0) + dh
    return clamp_box([x1,y1,x2,y2], W, H)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def sample_negatives(W, H, gt_boxes, n_samples, iou_thr=0.2):
    samples = []
    tries = 0
    max_tries = n_samples * 50
    win_w, win_h = 64, 128
    while len(samples) < n_samples and tries < max_tries:
        tries += 1
        # position aléatoire (garantit que la fenêtre est dans l'image)
        x1 = random.randint(0, max(0, W - win_w))
        y1 = random.randint(0, max(0, H - win_h))
        x2 = x1 + win_w
        y2 = y1 + win_h
        box = [x1,y1,x2,y2]
        ok = True
        for g in gt_boxes:
            if iou(box, g) > iou_thr:
                ok = False
                break
        if ok:
            samples.append(box)
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Dossier images")
    ap.add_argument("--labels", required=True, help="Dossier labels YOLO")
    ap.add_argument("--out_pos", required=True, help="Sortie patches positifs")
    ap.add_argument("--out_neg", required=True, help="Sortie patches négatifs")
    ap.add_argument("--neg_ratio", type=int, default=3)
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="Garder seulement ces id de classes (ex: 0). Si None: toutes.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(args.out_pos, exist_ok=True)
    os.makedirs(args.out_neg, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(args.images, "*.*")))
    count_pos = 0
    count_neg = 0

    for ip in img_paths:
        name = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(args.labels, name + ".txt")
        if not os.path.isfile(lp):
            continue
        try:
            img = Image.open(ip).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        gt_boxes = []
        with open(lp, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if (args.classes is not None) and (cls not in args.classes):
                continue
            cx = float(parts[1]); cy = float(parts[2])
            w  = float(parts[3]); h  = float(parts[4])
            box = yolo_to_xyxy((cx,cy,w,h), W, H)
            box = pad_to_ratio(box, target_ratio=64/128, W=W, H=H)
            gt_boxes.append(box)
            # crop + resize
            x1,y1,x2,y2 = map(int, box)
            crop = img.crop((x1,y1,x2,y2)).resize((64,128), Image.BILINEAR)
            outp = os.path.join(args.out_pos, f"{name}_pos_{count_pos:06d}.png")
            crop.save(outp)
            count_pos += 1

        # Negatives
        if gt_boxes:
            n_neg = args.neg_ratio * len(gt_boxes)
        else:
            # s'il n'y a pas d'annotations on peut en échantillonner quelques-uns
            n_neg = 2
        neg_boxes = sample_negatives(W, H, gt_boxes, n_neg, iou_thr=0.2)
        for box in neg_boxes:
            x1,y1,x2,y2 = map(int, box)
            crop = img.crop((x1,y1,x2,y2)).resize((64,128), Image.BILINEAR)
            outn = os.path.join(args.out_neg, f"{name}_neg_{count_neg:06d}.png")
            crop.save(outn)
            count_neg += 1

    print(f"Fini. Positifs: {count_pos}, Négatifs: {count_neg}")

if __name__ == "__main__":
    main()

'''
python ~/links/scratch/pieton_tm/scripts/patch_and_negatifs_inria.py   --images /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/images/   --labels /home/ousman/links/scrat
ch/pieton_tm/INRIA_Pedestrian/train/labels   --out_pos /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/positifs   --out_neg /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/train/negatifs   --neg_ratio 3 --classes 0
'''