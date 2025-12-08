#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import cv2
import argparse
from pathlib import Path

def natural_key(p: Path):
    m = re.search(r'(\d+)(?=\.[^.]+$)', p.name)  # extrait le numéro avant .jpg
    return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser(description="Concatène V*.jpg en MP4 à 30 fps")
    ap.add_argument("--in_dir", required=True, help="Dossier des images (ex: ...\\processed_train\\set00\\V000)")
    ap.add_argument("--out", required=True, help="Chemin de sortie MP4 (ex: ...\\processed_train\\set00\\V000.mp4)")
    ap.add_argument("--pattern", default="V*.jpg", help="Motif d'images (défaut: V*.jpg)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_mp4 = Path(args.out)
    if not in_dir.is_dir():
        raise SystemExit(f"[ERREUR] Dossier introuvable: {in_dir}")

    fps = 7.0  # forcé

    # Liste des images
    frames = sorted(in_dir.glob(args.pattern), key=natural_key)
    frames = [f for f in frames if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not frames:
        raise SystemExit(f"[ERREUR] Aucune image trouvée dans {in_dir}")

    # Taille des images
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise SystemExit(f"[ERREUR] Impossible de lire {frames[0]}")
    h, w = first.shape[:2]

    # Writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("[ERREUR] Impossible d’ouvrir le writer vidéo")

    # Ajout des images
    count = 0
    for img_path in frames:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Lecture échouée: {img_path.name}")
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        writer.write(img)
        count += 1

    writer.release()
    print(f"Vidéo écrite: {out_mp4} ({count} frames, {fps} fps, {w}x{h})")

if __name__ == "__main__":
    main()

#python .\images_to_videos.py --in_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\kitti_tracking\testing\image_02\0028" --out "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\kitti_tracking\testing\Videos\V0028.mp4" --pattern "*.png"