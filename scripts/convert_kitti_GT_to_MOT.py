import os
import argparse


def convert_dir(gt_dir, out_dir, keep_class="Pedestrian"):
    os.makedirs(out_dir, exist_ok=True)

    print("Conversion KITTI GT → MOT")
    print(f"Source      : {gt_dir}")
    print(f"Destination : {out_dir}")
    print(f"Classe gardée : {keep_class}")

    for file in sorted(os.listdir(gt_dir)):
        if not file.endswith(".txt"):
            continue

        in_path = os.path.join(gt_dir, file)
        out_path = os.path.join(out_dir, file)

        with open(in_path, "r") as fin, open(out_path, "w") as fout:
            for line in fin:
                p = line.strip().split()
                if len(p) < 10:
                    continue

                frame = int(p[0])
                track_id = int(p[1])
                obj_type = p[2]  # Pedestrian, Car, DontCare, ...

                # ignore les objets sans identité
                if track_id == -1:
                    continue

                # garde seulement la classe voulue
                if obj_type != keep_class:
                    continue

                x1 = float(p[6])
                y1 = float(p[7])
                x2 = float(p[8])
                y2 = float(p[9])

                w = x2 - x1
                h = y2 - y1

                # Format MOT : frame,id,x,y,w,h,-1,-1,-1
                fout.write(f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")

        print(f"[OK] {file} → {out_path}")

    print("\nConversion terminée.")


def main():
    parser = argparse.ArgumentParser(
        description="Convertir les GT KITTI Tracking en format MOT."
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Dossier d'entrée avec les GT KITTI (label_02)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Dossier de sortie pour les fichiers MOT",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Pedestrian",
        help="Classe à garder (Pedestrian, Car, Cyclist, ...). Défaut: Pedestrian",
    )

    args = parser.parse_args()
    convert_dir(args.gt_dir, args.out_dir, keep_class=args.class_name)


if __name__ == "__main__":
    main()

# python convert_kitti_gt_to_mot.py --gt_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\kitti_tracking\training\label_02" --out_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\kitti_tracking\training\label_02_MOT"