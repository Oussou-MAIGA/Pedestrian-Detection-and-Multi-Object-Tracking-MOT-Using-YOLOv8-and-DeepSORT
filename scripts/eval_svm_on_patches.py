#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_svm_on_patches.py
Évalue un SVM linéaire (features Haar) sur les patchs 64x128 déjà extraits.

Entrées:
  --modele   chemin vers .pkl/.joblib (LinearSVC sauvegardé avec joblib)
  --racine   racine des patchs (contient {positifs_qualite,negatifs_qualite} ou {positives,negatives})
  --set      setXX (obligatoire)
  --video    VYYY (optionnel, sinon toutes les vidéos du set)
  --seuil_decision   seuil sur decision_function (défaut 0.0)
  --taille   LxH (défaut 64x128)
  --ondelette haar (défaut)
  --sorties  dossier sortie pour les rapports

Sorties:
  resume.json + resume.txt dans --sorties
"""

import argparse, json
from pathlib import Path
import numpy as np
import cv2, pywt, joblib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score

# ---------- features ----------
def zscore(img):
    x = img.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

def desc_haar(image, L=64, H=128, wave="haar"):
    if image.shape[:2] != (H, L):
        image = cv2.resize(image, (L, H), interpolation=cv2.INTER_LINEAR)
    X = zscore(image)
    _, (LH, HL, HH) = pywt.dwt2(X, wave)
    f = np.concatenate([LH.ravel(), HL.ravel(), HH.ravel()]).astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-8)

# ---------- utils ----------
EXTS = (".png",".jpg",".jpeg",".PNG",".JPG",".JPEG")

def list_imgs(dir_path: Path):
    return sorted([p for p in dir_path.rglob("*") if p.suffix in EXTS])

def pick_class_dirs(racine: Path):
    # priorité aux versions "qualité", sinon fallback
    posq, negq = racine/"positifs_qualite", racine/"negatifs_qualite"
    
    if posq.is_dir() and negq.is_dir():
        return posq, negq

    return None, None

def subset_by_set_video(base_dir: Path, set_name: str, video_name: str|None):
    root = base_dir / set_name
    if not root.is_dir():
        return []
    if video_name:
        cand = root / video_name
        return list_imgs(cand) if cand.is_dir() else []
    # toutes les vidéos du set
    out = []
    for d in sorted(root.glob("V*")):
        if d.is_dir():
            out.extend(list_imgs(d))
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Évaluation SVM sur patchs (set obligatoire, vidéo optionnelle)")
    ap.add_argument("--modele",   type=Path, required=True)
    ap.add_argument("--racine",   type=Path, required=True, help="racine des patchs (dataset_caltech/train)")
    ap.add_argument("--set",      required=True, help="ex: set00")
    ap.add_argument("--video",    default="",   help="ex: V000 (optionnel)")
    ap.add_argument("--seuil_decision", type=float, default=0.0)
    ap.add_argument("--taille",   default="64x128")
    ap.add_argument("--ondelette",default="haar")
    ap.add_argument("--sorties",  type=Path, required=True)
    args = ap.parse_args()

    # taille
    try:
        L, H = [int(t) for t in args.taille.lower().split("x")]
    except:
        raise SystemExit("[ERREUR] --taille doit être de la forme LxH, ex: 64x128")

    # sorties
    args.sorties.mkdir(parents=True, exist_ok=True)

    # répertoires classes
    pos_dir, neg_dir = pick_class_dirs(args.racine)
    if pos_dir is None:
        raise SystemExit(f"[ERREUR] Impossible de trouver les dossiers 'positifs_qualite/negatifs_qualite' sous: "
                         f"{args.racine}")

    # sélection set / vidéo
    video_name = args.video if args.video else None
    pos_paths = subset_by_set_video(pos_dir, args.set, video_name)
    neg_paths = subset_by_set_video(neg_dir, args.set, video_name)

    print(f"[INFO] set={args.set}  video={args.video or 'ALL'}")
    print(f"[INFO] pos trouvés: {len(pos_paths)} | neg trouvés: {len(neg_paths)}")

    if len(pos_paths) == 0 and len(neg_paths) == 0:
        raise SystemExit("[ERREUR] Aucun patch trouvé. Vérifie --racine / --set / --video.")

    # charge modèle
    pack = joblib.load(args.modele)
    svm  = pack["svm"]
    wave = args.ondelette

    y_true, scores, y_pred = [], [], []

    def process_batch(paths, label):
        for i, p in enumerate(paths, 1):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            f = desc_haar(im, L, H, wave)
            sc = float(svm.decision_function(f.reshape(1, -1))[0])
            pred = 1 if sc >= args.seuil_decision else -1
            y_true.append(label); scores.append(sc); y_pred.append(pred)
            if i % 500 == 0:
                print(f"  … {i} / {len(paths)} ({'pos' if label==1 else 'neg'})")

    print("[INFO] Évaluation des positifs…")
    process_batch(pos_paths, +1)
    print("[INFO] Évaluation des négatifs…")
    process_batch(neg_paths, -1)

    y_true = np.array(y_true, dtype=np.int8)
    y_pred = np.array(y_pred, dtype=np.int8)
    scores = np.array(scores, dtype=np.float32)

    if y_true.size == 0:
        raise SystemExit("[ERREUR] Rien n’a été évalué.")

    # métriques
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average='binary', pos_label=1, zero_division=0
    )
    try:
        # AUC nécessitera des deux classes présentes
        y_bin = (y_true == 1).astype(np.int32)
        auc = roc_auc_score(y_bin, scores)
    except Exception:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])  # [[TP FN],[FP TN]] si labels=[1,-1]
    TP = int(cm[0,0]); FN = int(cm[0,1]); FP = int(cm[1,0]); TN = int(cm[1,1])

    # sauvegardes
    resume = {
        "set": args.set,
        "video": args.video or "ALL",
        "nb_pos": len(pos_paths), "nb_neg": len(neg_paths),
        "seuil_decision": args.seuil_decision,
        "taille": [L, H], "ondelette": wave,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auc": auc,
    }
    (args.sorties/"resume.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"set={args.set}  video={args.video or 'ALL'}")
    lines.append(f"pos={len(pos_paths)}  neg={len(neg_paths)}  seuil={args.seuil_decision}")
    lines.append(f"taille={L}x{H}  ondelette={wave}")
    lines.append(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    lines.append(f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    lines.append(f"auc={auc:.4f}" if auc is not None else "auc=NA (classes insuffisantes)")
    (args.sorties/"resume.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== RÉSUMÉ ===")
    for L_ in lines:
        print(L_)
    print(f"\n[OK] Résultats → {args.sorties/'resume.json'}")

if __name__ == "__main__":
    main()