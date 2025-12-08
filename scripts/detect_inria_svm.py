
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détection sliding-window + NMS avec SVM (features ondelette Haar), et sauvegarde:
 - JSON par image (boxes + scores)
 - Images annotées (--draw)
 - Planches/grilles type YOLO (--grid_batch, --grid_cols, --grid_tile_w/h)

Usage (exemple):
python detect_inria_svm.py \
  --images_dir /path/INRIA/Test/images \
  --model_path /path/modeles/svm/svm_haar_64x128_inria.joblib \
  --out_dir    /path/sorties_detect/inria_test \
  --stride 16 --nms_iou 0.5 --scales 1.0,0.9,0.8,0.7,0.6,0.5 \
  --draw \
  --grid_batch 16 --grid_cols 4 --grid_tile_w 320 --grid_tile_h 240
"""
import os, argparse, glob, json, math
from PIL import Image, ImageDraw
import numpy as np
from joblib import load
import pywt

def make_thumbnail(canvas_w, canvas_h, img):
    
    w, h = img.size
    ratio = min(canvas_w / float(w), canvas_h / float(h))
    nw = max(1, int(w * ratio))
    nh = max(1, int(h * ratio))
    imr = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    ox = (canvas_w - nw) // 2
    oy = (canvas_h - nh) // 2
    canvas.paste(imr, (ox, oy))
    return canvas

def wavelet_energy_grid(img_gray, levels=3, grid=(8,4), wavelet='haar'):
    H, W = img_gray.shape
    gy, gx = grid
    cell_h = H // gy
    cell_w = W // gx
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=levels)
    feats = []
    for lev in range(1, levels+1):
        (LH, HL, HH) = coeffs[lev]
        for band in (LH, HL, HH):
            bimg = Image.fromarray(np.float32(np.abs(band)))
            bimg = bimg.resize((W, H), Image.BILINEAR)
            b = np.asarray(bimg, dtype=np.float32)
            for iy in range(gy):
                for ix in range(gx):
                    y1 = iy*cell_h; y2 = (iy+1)*cell_h if iy<gy-1 else H
                    x1 = ix*cell_w; x2 = (ix+1)*cell_w if ix<gx-1 else W
                    patch = b[y1:y2, x1:x2]
                    feats.append(np.sum(patch*patch, dtype=np.float64))
    return np.array(feats, dtype=np.float32)

def nms(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def detect_on_image(img_pil, model, scales, stride, min_wh=(64,128), score_min=-1e9):
    W0, H0 = img_pil.size
    all_boxes, all_scores = [], []
    for s in scales:
        W = int(round(W0 * s)); H = int(round(H0 * s))
        if W < min_wh[0] or H < min_wh[1]:
            break
        imr = img_pil.resize((W, H), Image.BILINEAR).convert("L")
        arr = np.asarray(imr, dtype=np.float32)
        win_w, win_h = 64, 128
        for y in range(0, H - win_h + 1, stride):
            for x in range(0, W - win_w + 1, stride):
                patch = arr[y:y+win_h, x:x+win_w]
                feats = wavelet_energy_grid(patch, levels=3, grid=(8,4), wavelet='haar')[None, :]
                score = float(model.decision_function(feats)[0])
                if score >= score_min:
                    x1 = int(x / s); y1 = int(y / s)
                    x2 = int((x + win_w) / s); y2 = int((y + win_h) / s)
                    all_boxes.append([x1,y1,x2,y2])
                    all_scores.append(score)
    return all_boxes, all_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--score_min", type=float, default=-1.0)
    ap.add_argument("--draw", action="store_true", help="Enregistre des PNG annotés (un fichier par image)")
    ap.add_argument("--scales", type=str, default="1.0,0.9,0.8,0.7,0.6,0.5")
    # Grille type YOLO (simplifiée): une seule option --grid_batch
    ap.add_argument("--grid_batch", type=int, default=0, help="Nb d'images par planche (0=off, ex:16)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grids_dir = os.path.join(args.out_dir, "grids")
    if args.grid_batch and args.grid_batch > 0:
        os.makedirs(grids_dir, exist_ok=True)
    # Auto grid params (like YOLO): infer cols/rows from grid_batch; fixed tile size
    tile_w, tile_h = 320, 240
    if args.grid_batch and args.grid_batch > 0:
        # choose cols as largest divisor <= sqrt(n) for nice layout
        n = max(1, args.grid_batch)
        root = int(math.sqrt(n))
        auto_grid_cols = None
        for c in range(root, 0, -1):
            if n % c == 0:
                auto_grid_cols = c
                break
        if auto_grid_cols is None:
            auto_grid_cols = root
    else:
        auto_grid_cols = 4


    model = load(args.model_path)
    img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.*")))
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    page_images = []
    page_idx = 0

    for ip in img_paths:
        try:
            img = Image.open(ip).convert("RGB")
        except Exception:
            continue
        boxes, scores = detect_on_image(img, model, scales, args.stride, score_min=args.score_min)
        keep = nms(boxes, scores, iou_thr=args.nms_iou)
        boxes_nms = [boxes[i] for i in keep]
        scores_nms = [float(scores[i]) for i in keep]

        name = os.path.splitext(os.path.basename(ip))[0]
        with open(os.path.join(args.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"boxes": boxes_nms, "scores": scores_nms, "image": os.path.basename(ip)}, f)

        annotated = None
        if args.draw or (args.grid_batch and args.grid_batch > 0):
            # Crée une copie pour ne pas altérer l'original dans la boucle
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)
            for (x1,y1,x2,y2), sc in zip(boxes_nms, scores_nms):
                draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
                draw.text((x1, max(0,y1-12)), f"{sc:.2f}", fill=(255,0,0))

        if args.draw and annotated is not None:
            annotated.save(os.path.join(args.out_dir, f"{name}_det.png"))

        if args.grid_batch and args.grid_batch > 0 and annotated is not None:
            tile = make_thumbnail(tile_w, tile_h, annotated)
            page_images.append(tile)
            if len(page_images) >= args.grid_batch:
                cols = max(1, auto_grid_cols)
                rows = int(np.ceil(len(page_images)/float(cols)))
                W = cols * tile_w
                H = rows * tile_h
                page = Image.new('RGB', (W, H), (255,255,255))
                for idx, im in enumerate(page_images):
                    r = idx // cols; c = idx % cols
                    page.paste(im, (c*tile_w, r*tile_h))
                page_idx += 1
                page.save(os.path.join(grids_dir, f'page_{page_idx:04d}.png'))
                page_images = []

    # flush final
    if args.grid_batch and args.grid_batch > 0 and len(page_images) > 0:
        cols = max(1, auto_grid_cols)
        rows = int(np.ceil(len(page_images)/float(cols)))
        W = cols * tile_w
        H = rows * tile_h
        page = Image.new('RGB', (W, H), (255,255,255))
        for idx, im in enumerate(page_images):
            r = idx // cols; c = idx % cols
            page.paste(im, (c*tile_w, r*tile_h))
        page_idx += 1
        page.save(os.path.join(grids_dir, f'page_{page_idx:04d}.png'))

    print("Détection terminée. JSON dans", args.out_dir)
    if args.grid_batch and args.grid_batch > 0:
        print("Planches sauvegardées dans", grids_dir)

if __name__ == "__main__":
    main()
