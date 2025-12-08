import os
import cv2
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort


def box_iou_xyxy(b1, b2):
    """
    IoU entre deux boxes au format [x1, y1, x2, y2].
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter <= 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def main(args):
    # ============ CHEMINS VENANT DES ARGUMENTS ===================
    IMG_DIR = args.img_dir
    DETS_DIR = args.dets_dir
    OUT_DIR = args.out_dir

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)

    # ============ INITIALISATION REID =============
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
        embedder=args.embedder,   # modèle ReID
        half=False
    )

    # ============ LECTURE FRAMES ===================
    frames = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".png") or f.lower().endswith(".jpg")
    )

    if not frames:
        raise RuntimeError("Aucune image trouvée dans IMG_DIR")

    # pour convertir YOLO → pixels
    sample = cv2.imread(os.path.join(IMG_DIR, frames[0]))
    if sample is None:
        raise RuntimeError("Impossible de lire la première image.")
    H, W = sample.shape[:2]

    # ============ TRACKING =========================
    for frame_idx, fname in enumerate(frames):
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Impossible de lire {img_path}, on saute.")
            continue

        # fichier de detection correspondant
        base = os.path.splitext(fname)[0]
        det_path = os.path.join(DETS_DIR, base + ".txt")

        detections = []

        if os.path.isfile(det_path):
            with open(det_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls, cx, cy, w, h, conf = parts
                    cx, cy, w, h, conf = map(float, [cx, cy, w, h, conf])

                    # YOLO → pixels
                    x1 = (cx - w/2) * W
                    y1 = (cy - h/2) * H
                    bw = w * W
                    bh = h * H

                    detections.append([[x1, y1, bw, bh], conf, int(cls)])

        # --- PATCH MINIMAL : filtrer les bbox invalides (évite crop vide) ---
        filtered = []
        for box, conf, cls in detections:
            x1, y1, bw, bh = box

            # largeur/hauteur trop petites ou négatives
            if bw <= 1 or bh <= 1:
                continue
            # coordonnées hors image
            if x1 >= W or y1 >= H:
                continue
            if x1 + bw <= 0 or y1 + bh <= 0:
                continue

            filtered.append([box, conf, cls])

        detections = filtered
        # ---------------------------------------------------------------

        # --- tracking ---
        tracks = tracker.update_tracks(detections, frame=img)

        # image annotée
        vis = img.copy()

        # fichier label pour cette frame
        out_label_path = os.path.join(OUT_DIR, "labels", base + ".txt")
        ftxt = open(out_label_path, "w")

        for t in tracks:
            # 1) seulement les tracks confirmés
            if not t.is_confirmed():
                continue

            # 2) on affiche seulement si YOLO l'a mis à jour dans CETTE frame
            #    (sinon c'est juste la prédiction Kalman)
            if t.time_since_update > 0:
                continue

            tid = t.track_id
            l, ttop, r, b = t.to_ltrb()
            w_box = r - l
            h_box = b - ttop

            # ========= récupérer conf + classe façon YOLO =========
            # On cherche la détection avec la plus grande IoU
            best_iou = 0.0
            best_conf = None
            best_cls = None

            for box, conf, cls in detections:
                dx1, dy1, bw, bh = box
                dx2 = dx1 + bw
                dy2 = dy1 + bh
                det_box_xyxy = [dx1, dy1, dx2, dy2]
                trk_box_xyxy = [l, ttop, r, b]

                iou = box_iou_xyxy(trk_box_xyxy, det_box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = conf
                    best_cls = cls

            # Si on ne trouve rien, on met une valeur par défaut
            if best_conf is None:
                best_conf = 1.0
                best_cls = 0  # personne

            # Nom de classe 
            if best_cls == 0:
                class_name = "person"
            else:
                class_name = f"class{int(best_cls)}"

            label_text = f"{class_name} {best_conf:.2f} id:{tid}"

            # dessin bbox + label style YOLO
            cv2.rectangle(vis, (int(l), int(ttop)), (int(r), int(b)), (255, 0, 0), 2)
            cv2.putText(vis, label_text, (int(l), int(ttop) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # conversion en YOLO-normalized pour sauvegarde
            cx_n = ((l + r) / 2.0) / W
            cy_n = ((ttop + b) / 2.0) / H
            w_n = (w_box) / W
            h_n = (h_box) / H

            # format YOLO = (classe tid cx cy w h)
            # on met classe=best_cls (0 = person) pour rester cohérent
            ftxt.write(f"{int(best_cls)} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f} {tid}\n")

        ftxt.close()

        # sauvegarde image annotée
        cv2.imwrite(os.path.join(OUT_DIR, fname), vis)

    print("Terminé ! Résultats dans :", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracking DeepSORT + ReID (mobilenet) à partir de labels YOLO."
    )

    parser.add_argument("--img_dir", required=True, type=str,
                        help="Dossier des images (frames)")
    parser.add_argument("--dets_dir", required=True, type=str,
                        help="Dossier des labels YOLO (cx cy w h conf)")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Dossier de sortie (images + labels trackés)")

    parser.add_argument("--embedder", default="mobilenet", type=str,
                        help="Modèle ReID (mobilenet par défaut)")
    parser.add_argument("--max_age", default=10, type=int,
                        help="max_age DeepSORT")
    parser.add_argument("--n_init", default=3, type=int,
                        help="n_init DeepSORT")
    parser.add_argument("--max_cosine_distance", default=0.4, type=float,
                        help="max_cosine_distance DeepSORT")

    args = parser.parse_args()
    main(args)