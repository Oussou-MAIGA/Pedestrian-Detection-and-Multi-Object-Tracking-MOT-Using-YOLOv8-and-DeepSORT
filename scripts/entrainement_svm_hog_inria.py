#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entrainement_svm_hog_inria.py
Apprend un Pipeline StandardScaler + LinearSVC sur HOG (64x128) et réalise une CV 5-fold
pour retenir AP_val, F1_best_val et threshold_F1_best_val. Sauvegarde le modèle et les courbes.

Exemple :
python entrainement_svm_hog_inria.py \  --features_dir /home/ousman/links/scratch/pieton_tm/gabarit/features_hog_inria_64x128 \  --models_dir   /home/ousman/links/scratch/pieton_tm/modeles/svm \  --results_dir  /home/ousman/links/scratch/pieton_tm/sorties_detect/svm/hog_eval \  --modele_nom   svm_hog_64x128_inria.joblib
"""
import argparse, os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_curves(thr, P, R, F1, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(); plt.plot(thr, P); plt.xlabel("Seuil (score)"); plt.ylabel("Precision"); plt.title(f"P_curve ({prefix})")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"P_curve_{prefix}.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(thr, R); plt.xlabel("Seuil (score)"); plt.ylabel("Recall"); plt.title(f"R_curve ({prefix})")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"R_curve_{prefix}.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(thr, F1); plt.xlabel("Seuil (score)"); plt.ylabel("F1"); plt.title(f"F1_curve ({prefix})")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"F1_curve_{prefix}.png"), dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--modele_nom", default="svm_hog_64x128_inria.joblib")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    X = np.load(os.path.join(args.features_dir, "X.npy"))
    y = np.load(os.path.join(args.features_dir, "y.npy"))

    pipe = make_pipeline(StandardScaler(with_mean=False),
                         LinearSVC(C=args.C, max_iter=args.max_iter, class_weight="balanced"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_all, y_all = [], []
    for tr, te in cv.split(X, y):
        pipe.fit(X[tr], y[tr])
        s = pipe.decision_function(X[te])
        scores_all.append(s); y_all.append(y[te])
    scores_all = np.concatenate(scores_all)
    y_all = np.concatenate(y_all)

    P, R, thr = precision_recall_curve(y_all, scores_all)
    F1 = 2*P*R/(P+R+1e-12)
    ap = float(average_precision_score(y_all, scores_all))
    thr_full = np.r_[thr, thr[-1] if len(thr)>0 else 0.0]
    best_idx = int(np.nanargmax(F1))
    f1_best = float(F1[best_idx]); thr_best = float(thr_full[best_idx])

    pipe.fit(X, y)

    import joblib
    joblib.dump(pipe, os.path.join(args.models_dir, args.modele_nom))
    plot_curves(thr_full, P, R, F1, args.results_dir, "val")

    rapport = {
        "AP_val": ap,
        "F1_best_val": f1_best,
        "threshold_F1_best_val": thr_best,
        "N": int(len(y)),
        "C": args.C, "max_iter": args.max_iter,
        "feature": "HOG 64x128 (cell 8x8, block 2x2, bins 9)"
    }
    with open(os.path.join(args.results_dir, "rapport_val.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print(f"AP_val={ap:.4f} | F1-opt={f1_best:.4f} @ seuil={thr_best:.4f}")
    print(f"Modèle sauvegardé: {os.path.join(args.models_dir, args.modele_nom)}")
    print(f"Courbes/rapport:   {args.results_dir}")

if __name__ == "__main__":
    main()

# ----------------------
# Commande exemple :
# python entrainement_svm_hog_inria.py \#   --features_dir /home/ousman/links/scratch/pieton_tm/gabarit/features_hog_inria_64x128 \#   --models_dir   /home/ousman/links/scratch/pieton_tm/modeles/svm \#   --results_dir  /home/ousman/links/scratch/pieton_tm/sorties_detect/svm/hog_eval \#   --modele_nom   svm_hog_64x128_inria.joblib
# ----------------------
