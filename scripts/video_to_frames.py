import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Découper une vidéo en images costco_1.jpg, costco_2.jpg, ...")
    parser.add_argument("--video_path", type=str, required=True, help="Chemin vers la vidéo .avi")
    parser.add_argument("--out_dir", type=str, required=True, help="Dossier de sortie pour les images")
    parser.add_argument("--prefix", type=str, default="costco_", help="Préfixe pour les images (par défaut: costco_)")
    parser.add_argument("--start_idx", type=int, default=1, help="Index de départ (par défaut: 1)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo {args.video_path}")

    idx = args.start_idx
    nb = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ex: costco_1.jpg, costco_2.jpg, ...
        fname = f"{args.prefix}{idx}.jpg"
        out_path = os.path.join(args.out_dir, fname)
        cv2.imwrite(out_path, frame)
        idx += 1
        nb += 1

    cap.release()
    print(f"Terminé : {nb} images sauvegardées dans {args.out_dir}")

if __name__ == "__main__":
    main()