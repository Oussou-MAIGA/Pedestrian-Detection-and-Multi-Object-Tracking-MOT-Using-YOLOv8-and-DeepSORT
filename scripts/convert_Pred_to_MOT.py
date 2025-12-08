import os
import glob
import argparse
import cv2


def convert_sequence(labels_dir, images_dir, out_file):
    """
    Convertit les fichiers YOLO tracking (6 colonnes) en format MOT.
    Format YOLO track :
        class xc yc w h track_id
    Format MOT attendu :
        frame,id,x,y,w,h,-1,-1,-1
    """

    # --- 1. Récupérer la taille des images (KITTI) ---
    sample_img = sorted(glob.glob(os.path.join(images_dir, "*.*")))[0]
    img = cv2.imread(sample_img)
    H, W = img.shape[:2]
    print(f"Taille image détectée : {W}x{H}")

    # --- 2. Fichiers labels (un .txt par frame) ---
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not label_files:
        print("Aucun .txt trouvé dans :", labels_dir)
        return

    with open(out_file, "w") as fout:
        for fpath in label_files:
            frame_str = os.path.basename(fpath).split(".")[0]  # "000000"
            frame = int(frame_str)

            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue

                    cls, xc, yc, w, h, tid = parts
                    tid = int(tid)

                    # YOLO normalized -> pixels
                    xc = float(xc) * W
                    yc = float(yc) * H
                    w  = float(w) * W
                    h  = float(h) * H

                    x = xc - w/2
                    y = yc - h/2

                    fout.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")

    print(f"Conversion terminée -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Convertir prédictions YOLO tracking en MOT")
    parser.add_argument("--labels_dir", required=True, type=str,
                        help="Dossier contenant les fichiers .txt de YOLO tracking")
    parser.add_argument("--images_dir", required=True, type=str,
                        help="Dossier des images KITTI correspondantes (ex: image_02/0000)")
    parser.add_argument("--out_file", required=True, type=str,
                        help="Fichier MOT de sortie (ex: bytetrack_MOT/0000.txt)")

    args = parser.parse_args()

    convert_sequence(args.labels_dir, args.images_dir, args.out_file)


if __name__ == "__main__":
    main()
    
# python convert_Pred_to_MOT.py --labels_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\Caltech_Pedestrian\runs\kitti_eval\bytetrack\0010\labels" --images_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\kitti_tracking\training\image_02\0010" --out_file "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\Caltech_Pedestrian\runs\kitti_eval\bytetrack_MOT/0010.txt"
