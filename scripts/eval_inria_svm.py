#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation des détections (JSON) vs labels YOLO (cx,cy,w,h relatifs).
- Associe détections ↔ GT par IoU>=--iou_thr en greedy (style Pascal/VOC).
- Trie toutes les détections par score → PR, AP, P/R/F1 vs seuil, seuil F1-opt.
- Sauvegarde: PR_curve_test.png, P_curve_test.png, R_curve_test.png, F1_curve_test.png, rapport_test.json

Entrées:
  --detections_dir : JSON issus de detect_inria_svm.py (post-NMS)
  --labels_dir     : .txt YOLO (une ou plusieurs classes; toutes prises comme "piéton")
  --images_dir     : images test (pour tailles exactes)
  --out_dir        : dossier de sortie

Option:
  --iou_thr        : IoU pour matching (par défaut 0.5 → mAP50)

Le script est headless (matplotlib backend Agg).
"""
import os, argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def yolo_to_xyxy(rel, W, H):
    cx, cy, w, h = rel
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2.0; y1 = cy - h/2.0
    x2 = cx + w/2.0; y2 = cy + h/2.0
    return [x1,y1,x2,y2]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

def load_gt(labels_dir, images_dir):
    gts = {}
    img_sizes = {}
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    for ip in img_paths:
        name = os.path.splitext(os.path.basename(ip))[0]
        try:
            W,H = Image.open(ip).size
        except Exception:
            continue
        img_sizes[name] = (W,H)
        lp = os.path.join(labels_dir, name + ".txt")
        boxes = []
        if os.path.isfile(lp):
            with open(lp, "r", encoding="utf-8") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) != 5: continue
                    cx,cy,w,h = [float(x) for x in parts[1:]]
                    boxes.append(yolo_to_xyxy((cx,cy,w,h), W, H))
        gts[name] = boxes
    return gts, img_sizes

def average_precision(precision, recall):
    # VOC 2010+: intégrale par interpolation en escalier, précision rendue monotone
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return float(ap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gts, sizes = load_gt(args.labels_dir, args.images_dir)

    # Charger toutes les detections
    dets_all = []  # (img_name, score, [x1,y1,x2,y2])
    json_paths = sorted(glob.glob(os.path.join(args.detections_dir, "*.json")))
    for jp in json_paths:
        name = os.path.splitext(os.path.basename(jp))[0]
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)
        for box, sc in zip(d.get("boxes", []), d.get("scores", [])):
            dets_all.append((name, float(sc), box))

    # Trier par score décroissant
    dets_all.sort(key=lambda t: t[1], reverse=True)

    # Matching greedy (un GT ne peut matcher qu'une détection)
    gt_used = {k: np.zeros(len(v), dtype=bool) for k,v in gts.items()}
    TP = []; FP = []
    for name, sc, box in dets_all:
        gtb = gts.get(name, [])
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gtb):
            if gt_used[name][j]:  # déjà apparié
                continue
            i = iou(box, gb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= args.iou_thr and best_j >= 0:
            TP.append(1); FP.append(0)
            gt_used[name][best_j] = True
        else:
            TP.append(0); FP.append(1)

    TP = np.array(TP, dtype=np.int32)
    FP = np.array(FP, dtype=np.int32)
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    npos = int(sum(len(v) for v in gts.values()))
    precision = cum_TP / np.maximum(cum_TP + cum_FP, 1e-9)
    recall = cum_TP / max(npos, 1e-9)

    # PR + AP
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR (IoU≥{args.iou_thr})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "PR_curve_test.png"), dpi=200); plt.close()
    ap_val = average_precision(precision, recall)

    # P/R/F1 vs seuil (en balayant la liste triée)
    scores_sorted = [sc for _, sc, _ in dets_all]
    if len(scores_sorted) == 0:
        thresholds = np.array([0.0])
        P = np.array([0.0]); R = np.array([0.0]); F1 = np.array([0.0])
    else:
        thresholds = np.array(scores_sorted)
        P = precision
        R = recall
        eps = 1e-9
        F1 = 2 * P * R / (P + R + eps)

    plt.figure(); plt.plot(thresholds, P); plt.xlabel('Seuil (score)'); plt.ylabel('Precision'); plt.title('P_curve (test)')
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "P_curve_test.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(thresholds, R); plt.xlabel('Seuil (score)'); plt.ylabel('Recall'); plt.title('R_curve (test)')
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "R_curve_test.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(thresholds, F1); plt.xlabel('Seuil (score)'); plt.ylabel('F1'); plt.title('F1_curve (test)')
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "F1_curve_test.png"), dpi=200); plt.close()

    # seuil F1-opt + métriques
    if len(F1) > 0:
        i_best = int(np.argmax(F1))
        thr_best = float(thresholds[i_best])
    else:
        thr_best = 0.0

    # Recompte au seuil F1-opt (TP/FP/FN)
    TPb = 0; FPb = 0
    matched = {k: np.zeros(len(v), dtype=bool) for k,v in gts.items()}
    dets_thr = [(n, sc, b) for (n, sc, b) in dets_all if sc >= thr_best]
    for name, sc, box in dets_thr:
        gtb = gts.get(name, [])
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gtb):
            if matched[name][j]: continue
            i = iou(box, gb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= args.iou_thr and best_j >= 0:
            TPb += 1
            matched[name][best_j] = True
        else:
            FPb += 1
    FN = sum((~matched[k]).sum() for k in matched.keys())

    # Rapport
    rapport = {
        "npos": int(npos),
        "detections": int(len(dets_all)),
        "AP_test": float(ap_val),
        "F1_best_threshold": float(thr_best),
        "TP_at_best": int(TPb),
        "FP_at_best": int(FPb),
        "FN_at_best": int(FN),
        "IoU_eval": float(args.iou_thr)
    }
    with open(os.path.join(args.out_dir, "rapport_test.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print("AP_test=", rapport["AP_test"], " Seuil F1-opt=", rapport["F1_best_threshold"])
    print("Rapport et courbes enregistrés dans", args.out_dir)

if __name__ == "__main__":
    main()