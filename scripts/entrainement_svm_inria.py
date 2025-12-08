
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraîne un SVM linéaire sur les features Haar (sans bootstrapping).
- Charge X.npy, y.npy
- Standardise (z-score), option PCA (off par défaut)
- CV pour rapport: PR, AP, F1(seuil), matrices de confusion
- Entraîne sur l'ensemble complet puis sauvegarde le modèle
Sorties:
  modèle: svm_haar_64x128_inria.joblib (+ scaler.joblib, pca.joblib si utilisé)
  courbes: PR_curve.png, P_curve.png, R_curve.png, F1_curve.png,
           confusion_matrix.png, confusion_matrix_normalized.png
  rapport: rapport.json
"""
import os, argparse, json
import matplotlib
matplotlib.use('Agg')  # Headless backend
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, confusion_matrix, classification_report
from joblib import dump
import matplotlib.pyplot as plt

def plot_pr(y_true, scores, out_png):
    precision, recall, thresh = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR curve (AP={ap:.3f})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return precision, recall, thresh, ap


def plot_curves_vs_threshold(y_true, scores, out_p, out_r, out_f1):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve retourne:
    # - precision et recall de longueur N+1
    # - thresholds de longueur N
    # Pour tracer P/R/F1 en fonction du seuil, on coupe P et R à N.
    if thresholds is None or len(thresholds) == 0:
        thr = np.array([0.0])
        precision_cut = np.array([precision[-1] if len(precision) else 0.0])
        recall_cut = np.array([recall[-1] if len(recall) else 0.0])
    else:
        thr = thresholds
        precision_cut = precision[:-1]
        recall_cut = recall[:-1]

    eps = 1e-9
    f1 = 2 * precision_cut * recall_cut / (precision_cut + recall_cut + eps)

    # P_curve
    plt.figure()
    plt.plot(thr, precision_cut)
    plt.xlabel('Seuil (score)')
    plt.ylabel('Precision')
    plt.title('P_curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(out_p, dpi=200); plt.close()

    # R_curve
    plt.figure()
    plt.plot(thr, recall_cut)
    plt.xlabel('Seuil (score)')
    plt.ylabel('Recall')
    plt.title('R_curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(out_r, dpi=200); plt.close()

    # F1_curve
    plt.figure()
    plt.plot(thr, f1)
    plt.xlabel('Seuil (score)')
    plt.ylabel('F1')
    plt.title('F1_curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(out_f1, dpi=200); plt.close()

    if len(f1) > 0:
        i_best = int(np.argmax(f1))
        return thr[i_best], float(f1[i_best])
    return 0.0, 0.0
def cm_plots(y_true, scores, thr, out_cm, out_cmn):
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # plot brute
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha='center', va='center')
    plt.tight_layout(); plt.savefig(out_cm, dpi=200); plt.close()

    # normalisée par ligne
    with np.errstate(divide='ignore', invalid='ignore'):
        cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure()
    plt.imshow(cmn, interpolation='nearest')
    plt.title('Confusion Matrix (normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i,j), val in np.ndenumerate(cmn):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center')
    plt.tight_layout(); plt.savefig(out_cmn, dpi=200); plt.close()
    return cm, cmn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True, help="Dossier contenant X.npy, y.npy")
    ap.add_argument("--models_dir", required=True, help="Dossier pour sauvegarder le modèle")
    ap.add_argument("--results_dir", required=True, help="Dossier pour courbes/rapports")
    ap.add_argument("--modele_nom", default="svm_haar_64x128_inria.joblib")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    X = np.load(os.path.join(args.features_dir, "X.npy"))
    y = np.load(os.path.join(args.features_dir, "y.npy"))
    if X.ndim != 2:
        raise RuntimeError("X.npy doit être 2D (N, D).")
    # Pipeline: scaler + LinearSVC (scores via decision_function)
    pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("clf", LinearSVC(C=1.0, class_weight='balanced', max_iter=5000))])

    # Scores CV pour PR/F1/CM
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_predict(pipe, X, y, cv=cv, method='decision_function', n_jobs=-1)

    # PR + AP
    pr_png = os.path.join(args.results_dir, "PR_curve.png")
    precision, recall, thresh, ap = None, None, None, None
    precision, recall, thresh, ap = plot_pr(y, scores, pr_png)

    # Courbes vs seuil
    p_png = os.path.join(args.results_dir, "P_curve.png")
    r_png = os.path.join(args.results_dir, "R_curve.png")
    f_png = os.path.join(args.results_dir, "F1_curve.png")
    thr_best, f1_best = plot_curves_vs_threshold(y, scores, p_png, r_png, f_png)

    # Matrices de confusion au seuil F1-optimal
    cm_png = os.path.join(args.results_dir, "confusion_matrix.png")
    cmn_png = os.path.join(args.results_dir, "confusion_matrix_normalized.png")
    cm, cmn = cm_plots(y, scores, thr_best, cm_png, cmn_png)

    # Fit final sur tout le set et sauvegardes
    pipe.fit(X, y)
    model_path = os.path.join(args.models_dir, args.modele_nom)
    dump(pipe, model_path)

    # Rapport JSON
    rapport = {
        "mAP": float(ap) if ap is not None else None,
        "threshold_F1_best": float(thr_best),
        "F1_best": float(f1_best),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": [[float(x) for x in row] for row in cmn],
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "model_path": model_path,
    }
    with open(os.path.join(args.results_dir, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print("mAP=", rapport["mAP"])
    print("Seuil F1-opt=", rapport["threshold_F1_best"])
    print("Modèle sauvegardé:", model_path)
    print("Courbes et rapport dans:", args.results_dir)

if __name__ == "__main__":
    main()
