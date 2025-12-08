#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_inria_hog_svm.py
Détection multi-échelle avec HOG+SVM sur INRIA (images complètes).
Voir le bloc de commandes tout en bas pour l'usage.

Exemple (Validation, score=-1 pour laisser l'éval choisir le meilleur seuil) :
python detect_inria_hog_svm.py \  --images_dir /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/valid/images \  --model_path /home/ousman/links/scratch/pieton_tm/modeles/svm/svm_hog_64x128_inria.joblib \  --out_dir    /home/ousman/links/scratch/pieton_tm/sorties_detect/svm/hog_inria_val_detect \  --scales 0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4 \  --stride 8 --nms_iou 0.5 --score_min -1 --draw --display_prob
"""
import argparse, os, json, numpy as np, cv2, glob

def imread_color(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Impossible de lire {path}")
    return im

def pyramid_scales(scales_str):
    return [float(s) for s in scales_str.split(",")] if isinstance(scales_str, str) else list(scales_str)

def hog_descriptor():
    return cv2.HOGDescriptor(_winSize=(64,128), _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)

def compute_hog_patch(hog, gray_patch_64x128):
    return hog.compute(gray_patch_64x128).reshape(1,-1)

def iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter / (areaA + areaB - inter + 1e-12)

def nms(boxes, scores, iou_thr):
    idx = np.argsort(scores)[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]; keep.append(i)
        rest = idx[1:]
        suppress = []
        for j, k in enumerate(rest):
            if iou_xyxy(boxes[i], boxes[k]) >= iou_thr:
                suppress.append(j)
        idx = np.delete(rest, suppress)
    return keep

def draw_box(img, box, txt=None, color=(0,0,255)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if txt is not None:
        cv2.putText(img, txt, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scales", default="0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--score_min", type=float, default=-1.0)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--display_prob", action="store_true", help="affiche p=sigmoid(s) au lieu de s")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import joblib
    pipe = joblib.load(args.model_path)  # StandardScaler + LinearSVC
    hog = hog_descriptor()

    im_files = sorted(sum([glob.glob(os.path.join(args.images_dir, p)) for p in ("*.jpg","*.jpeg","*.png","*.bmp")], []))
    for path in im_files:
        img = imread_color(path); h0, w0 = img.shape[:2]
        gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        all_boxes, all_scores = [], []
        for s in pyramid_scales(args.scales):
            gray = cv2.resize(gray0, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_LINEAR)
            H, W = gray.shape
            for y in range(0, H-128+1, args.stride):
                for x in range(0, W-64+1, args.stride):
                    patch = gray[y:y+128, x:x+64]
                    feat = compute_hog_patch(hog, patch)
                    score = float(pipe.decision_function(feat)[0])
                    if score >= args.score_min:
                        x1, y1 = int(x/s), int(y/s)
                        x2, y2 = int((x+64)/s), int((y+128)/s)
                        all_boxes.append([x1,y1,x2,y2])
                        all_scores.append(score)

        if len(all_boxes) == 0:
            out = {"boxes": [], "scores": [], "probs": [], "meta":{"nms_iou":args.nms_iou, "eval_iou":0.5}}
            base = os.path.splitext(os.path.basename(path))[0]
            json_path = os.path.join(args.out_dir, base + ".json")
            with open(json_path, "w") as f:
                json.dump(out, f, indent=2)
            continue

        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)

        keep = nms(boxes, scores, args.nms_iou)
        boxes = boxes[keep]; scores = scores[keep]
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))  # pour affichage

        out = {
            "boxes": boxes.tolist(),
            "scores": [float(s) for s in scores],
            "probs":  [float(p) for p in probs],
            "meta":   {"nms_iou": args.nms_iou, "eval_iou": 0.5, "stride": args.stride, "scales": args.scales}
        }
        base = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(args.out_dir, base + ".json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

        if args.draw:
            vis = img.copy()
            for b, s, p in zip(boxes, scores, probs):
                txt = f"{p:.2f}" if args.display_prob else f"{s:.2f}"
                draw_box(vis, b, txt)
            out_img = os.path.join(args.out_dir, os.path.basename(path).rsplit(".",1)[0] + "_det.png")
            cv2.imwrite(out_img, vis)

    print(f"Détection terminée. JSON/PNGs dans {args.out_dir}")

if __name__ == "__main__":
    main()

# ----------------------
# Validation (laisser score=-1 pour balayage des seuils) :
# python ~/links/scratch/pieton_tm/scripts/detect_inria_hog_svm.py --images_dir /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/valid/images  --model_path /home/ousman/links/scratch/pieton_tm/modeles/svm/svm_hog_64x128_inria.joblib  --out_dir    /home/ousman/links/scratch/pieton_tm/sorties_detect/hog/inria_val/val_detect  --scales 0.3  --stride 6 --nms_iou 0.2 --score_min 0.1 --draw --display_prob
#
# Ensuite évaluer (mAP50, F1_opt) avec ton eval_inria_svm.py existant :
# python eval_inria_svm.py \#   --detections_dir /home/ousman/links/scratch/pieton_tm/sorties_detect/svm/hog_inria_val_detect \#   --labels_dir     /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/valid/labels \#   --images_dir     /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/valid/images \#   --out_dir        /home/ousman/links/scratch/pieton_tm/sorties_detect/svm_eval/hog_inria_val \#   --iou_thr        0.5
#
# Test final (réutiliser le seuil F1-opt trouvé sur Val, ex: 2.18) :
# python detect_inria_hog_svm.py \#   --images_dir /home/ousman/links/scratch/pieton_tm/INRIA_Pedestrian/test/images \#   --model_path /home/ousman/links/scratch/pieton_tm/modeles/svm/svm_hog_64x128_inria.joblib \#   --out_dir    /home/ousman/links/scratch/pieton_tm/sorties_detect/svm/hog_inria_test_detect \#   --scales 0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4 \#   --stride 8 --nms_iou 0.5 --score_min 2.18 --draw --display_prob
# ----------------------
