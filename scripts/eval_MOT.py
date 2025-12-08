import os
import argparse
import pandas as pd
import motmetrics as mm
import json


def read_mot(path):
    """Lit un fichier MOT (frame,id,x,y,w,h,...) en DataFrame.
       Si le fichier est vide -> retourne un DataFrame vide."""
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=['FrameId','Id','X','Y','W','H','A1','A2','A3'])

    df = pd.read_csv(path, header=None)
    df.columns = ['FrameId', 'Id', 'X', 'Y', 'W', 'H', 'A1', 'A2', 'A3']
    return df


def evaluate_pair(gt_path, pred_path):
    gt = read_mot(gt_path)
    pr = read_mot(pred_path)

    if gt.empty:
        print(f"[INFO] GT vide : {gt_path}, on saute.")
        return None

    acc = mm.MOTAccumulator(auto_id=True)

    for frame in sorted(gt.FrameId.unique()):
        gt_f = gt[gt.FrameId == frame]
        pr_f = pr[pred_frame_mask(pr, frame)]

        gt_ids = gt_f.Id.tolist()
        pr_ids = pr_f.Id.tolist()

        gt_boxes = gt_f[['X', 'Y', 'W', 'H']].values
        pr_boxes = pr_f[['X', 'Y', 'W', 'H']].values

        dists = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(gt_ids, pr_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['idf1', 'idp', 'idr', 'mota', 'motp', 'num_switches'],
        name='res'
    ).iloc[0]

    return {k: float(summary[k]) for k in summary.index}


def pred_frame_mask(pr, frame):
    # petit helper pour garder la même structure que pour gt
    return pr.FrameId == frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--save_dir", default="eval_results", help="Dossier de sauvegarde")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    print(f"GT   : {gt_dir}")
    print(f"PRED : {pred_dir}")
    print(f"SAVE : {save_dir}\n")

    seq_results = []
    metrics_names = ['idf1', 'idp', 'idr', 'mota', 'motp', 'num_switches']

    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue

        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.isfile(pred_path):
            print(f"[WARN] Pas de prédiction pour {fname}.")
            continue

        print(f"Évalue : {fname}")
        res = evaluate_pair(gt_path, pred_path)
        if res is None:
            continue

        seq_id = os.path.splitext(fname)[0]
        seq_results.append({"seq": seq_id, **res})

    if not seq_results:
        print("Aucun résultat valide.")
        return

    df = pd.DataFrame(seq_results)

    print("\n=== Résultats par séquence ===")
    print(df.to_string(index=False))

    # SAVE RESULTS PER SEQUENCE EN JSON
    per_seq_path = os.path.join(save_dir, "results_per_sequence.json")
    with open(per_seq_path, "w", encoding="utf-8") as f:
        json.dump(seq_results, f, indent=4)
    print(f"\n Sauvegardé : {per_seq_path}")

    # --- GLOBAL SUMMARY ---
    avg = {}
    for m in metrics_names:
        if m == "num_switches":
            avg[m] = float(df[m].sum())
        else:
            avg[m] = float(df[m].mean())

    print("\n=== Résumé global ===")
    for k, v in avg.items():
        if k == "num_switches":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")

    # SAVE GLOBAL SUMMARY EN JSON
    summary_path = os.path.join(save_dir, "summary_global.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=4)
    print(f"Sauvegardé : {summary_path}")


if __name__ == "__main__":
    main()
    
# python eval_MOT.py --gt_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\kitti_tracking\training\label_02_MOT" --pred_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\Caltech_Pedestrian\runs\kitti_eval\bytetrack_MOT" --save_dir "C:\Users\technicien\Downloads\UMoncton_Automne_2025_2026\Caltech_Pedestrian\runs\kitti_eval"