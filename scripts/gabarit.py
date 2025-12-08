#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construction d’un gabarit (template) Haar à partir des patches POSITIFS 64×128.

Entrées (arborescence) :
  ~/links/scratch/pieton_tm/
    dataset_caltech/train/positives/setXX/VYYY/pos_*.png

Sorties :
  ~/links/scratch/pieton_tm/gabarit/
    gabarit_haar_64x128.npy      ← le vecteur moyen (6144,)
    gabarit_haar_64x128.json     ← méta-données
    journal_gabarit.txt          ← journal horodaté (progression + stats)

Notes :
- DWT-2 (ondelette Haar, niveau 1)
- sous-bandes conservées : LH, HL, HH  → concaténées (6144 dim)
- normalisation : z-score par patch puis L2 par vecteur
- gabarit = moyenne des vecteurs, renormalisé L2
"""

from __future__ import annotations
import json, time, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import pywt
from collections import defaultdict

# ======== paramètres par défaut ========
LARGEUR_FENETRE = 64
HAUTEUR_FENETRE = 128
ONDELETTE = "haar"
PAS_JOURNAL = 500        # écrire une ligne de progression au journal toutes les N images
AFFICHAGE_TERMINAL = 1   # 0 = silencieux, 1 = sommaire, 2 = verbeux

# =============== utilitaires FS ===============
def creer_dossier(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def horodatage() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ecrire_ligne(journal: Path, texte: str, echo: int = 0):
    texte = f"[{horodatage()}] {texte}"
    with journal.open("a", encoding="utf-8") as f:
        f.write(texte + "\n")
    if echo and AFFICHAGE_TERMINAL >= echo:
        print(texte)

def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

# =============== descripteur Haar ===============
def descripteur_haar(img_gray: np.ndarray,
                     largeur: int = LARGEUR_FENETRE,
                     hauteur: int = HAUTEUR_FENETRE,
                     ondelette: str = ONDELETTE) -> np.ndarray:
    """Retourne le vecteur concaténé LH|HL|HH (LL ignoré), L2-normalisé."""
    if img_gray.shape != (hauteur, largeur):
        img_gray = cv2.resize(img_gray, (largeur, hauteur), interpolation=cv2.INTER_LINEAR)
    X = zscore(img_gray)
    LL, (LH, HL, HH) = pywt.dwt2(X, ondelette)
    feat = np.concatenate([LH.ravel(), HL.ravel(), HH.ravel()]).astype(np.float32)
    n = np.linalg.norm(feat) + 1e-8
    return feat / n

# =============== itération sur les patches ===============
def iter_patches_positifs(racine_positifs: Path,
                          set_filtre: str = "",
                          video_filtre: str = "") -> Tuple[Path, str, str]:
    """
    Itère tous les fichiers pos_*.png sous positives/(set)/(video).
    Rend (chemin_fichier, set_name, video_name).
    """
    sets = [racine_positifs / set_filtre] if set_filtre else sorted(racine_positifs.glob("set*"))
    for s in sets:
        if not s.is_dir():
            continue
        set_name = s.name
        videos = [s / video_filtre] if video_filtre else sorted(s.glob("V*"))
        for v in videos:
            if not v.is_dir():
                continue
            vid_name = v.name
            for p in sorted(v.glob("pos_*.png")):
                yield p, set_name, vid_name

# =============== principal ===============
def construire_gabarit(
    racine_pieton_tm: Path,
    largeur: int = LARGEUR_FENETRE,
    hauteur: int = HAUTEUR_FENETRE,
    ondelette: str = ONDELETTE,
    set_filtre: str = "",
    video_filtre: str = "",
    max_patches: int = 0
):
    # chemins d’E/S
    pos_root = racine_pieton_tm / "dataset_caltech" / "train" / "positives"
    if not pos_root.is_dir():
        raise SystemExit(f"[ERREUR] Dossier positifs introuvable: {pos_root}")

    out_dir = racine_pieton_tm / "gabarit"
    creer_dossier(out_dir)
    npy_path  = out_dir / f"gabarit_haar_{largeur}x{hauteur}.npy"
    meta_path = out_dir / f"gabarit_haar_{largeur}x{hauteur}.json"
    journal   = out_dir / "journal_gabarit.txt"

    # journal : entête
    ecrire_ligne(journal, "Démarrage construction du gabarit Haar", echo=1)
    ecrire_ligne(journal, f"Racine positifs : {pos_root}")
    ecrire_ligne(journal, f"Filtres         : set={set_filtre or 'tous'}, video={video_filtre or 'toutes'}")
    ecrire_ligne(journal, f"Paramètres      : taille={largeur}x{hauteur}, ondelette={ondelette}, max_patches={max_patches or 'tous'}")

    # passage 1 : compter pour progression + répartition
    total_eligibles = 0
    repartition_sets = defaultdict(int)
    repartition_videos = defaultdict(int)
    for _, set_name, vid_name in iter_patches_positifs(pos_root, set_filtre, video_filtre):
        total_eligibles += 1
        repartition_sets[set_name] += 1
        repartition_videos[f"{set_name}/{vid_name}"] += 1
        if max_patches and total_eligibles >= max_patches:
            break

    if total_eligibles == 0:
        ecrire_ligne(journal, "Aucun patch positif trouvé avec ces filtres.", echo=1)
        raise SystemExit("[ERREUR] Aucun patch positif.")

    ecrire_ligne(journal, f"Patches éligibles : {total_eligibles} (après filtres)", echo=1)
    if AFFICHAGE_TERMINAL >= 2:
        ecrire_ligne(journal, f"Répartition sets  : {dict(repartition_sets)}")
        # éviter de spammer le terminal, on mettra le détail des vidéos dans le JSON résumé.

    # passage 2 : accumulation des descripteurs
    feats = []
    nb_lus = 0
    t0 = time.time()
    for img_path, set_name, vid_name in iter_patches_positifs(pos_root, set_filtre, video_filtre):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        f = descripteur_haar(img, largeur, hauteur, ondelette)
        feats.append(f)
        nb_lus += 1

        # progression : on journalise toutes les PAS_JOURNAL images
        if nb_lus % PAS_JOURNAL == 0:
            pct = 100.0 * nb_lus / total_eligibles
            ecrire_ligne(journal, f"Progression {nb_lus}/{total_eligibles} ({pct:.1f}%)")

        if max_patches and nb_lus >= max_patches:
            break

    if not feats:
        ecrire_ligne(journal, "Aucun descripteur produit (lecture images ?).", echo=1)
        raise SystemExit("[ERREUR] Aucun descripteur produit.")

    # gabarit = moyenne, puis L2
    M = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)
    M /= (np.linalg.norm(M) + 1e-8)

    # sauvegardes
    np.save(str(npy_path), M)
    duree = time.time() - t0

    resume = {
        "type": "gabarit_haar",
        "largeur": largeur,
        "hauteur": hauteur,
        "ondelette": ondelette,
        "dim_vecteur": int(M.shape[0]),
        "nb_patches_utilises": int(len(feats)),
        "nb_patches_eligibles": int(total_eligibles),
        "filtre_set": set_filtre or "tous",
        "filtre_video": video_filtre or "toutes",
        "repartition_sets": dict(sorted(repartition_sets.items())),
        # pour ne pas surcharger, on ne met pas toutes les vidéos si c'est énorme :
        "repartition_videos_top10": sorted(repartition_videos.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "duree_secondes": round(duree, 3),
        "npy_path": str(npy_path)
    }
    meta_path.write_text(json.dumps(resume, indent=2), encoding="utf-8")

    # journal : fin
    ecrire_ligne(journal, f"Gabarit sauvegardé → {npy_path}", echo=1)
    ecrire_ligne(journal, f"Résumé méta        → {meta_path}")
    ecrire_ligne(journal, f"Vecteur            → dim={M.shape[0]} | patches_utilisés={len(feats)}")
    ecrire_ligne(journal, f"Durée totale       → {duree:.2f}s")
    if AFFICHAGE_TERMINAL >= 1:
        print(f"Gabarit: {npy_path.name} | dim={M.shape[0]} | patches={len(feats)}")

# ================== CLI ==================
def main():
    ap = argparse.ArgumentParser(description="Construire le gabarit (template) Haar depuis les POSITIFS (avec journal)")
    ap.add_argument("--pieton_tm", type=Path, required=True,
                    help="racine du projet (contient dataset_caltech/…/positives)")
    ap.add_argument("--largeur", type=int, default=LARGEUR_FENETRE)
    ap.add_argument("--hauteur", type=int, default=HAUTEUR_FENETRE)
    ap.add_argument("--ondelette", default=ONDELETTE)
    ap.add_argument("--set", default="", help="ex: set00 (vide = tous)")
    ap.add_argument("--video", default="", help="ex: V000 (vide = toutes)")
    ap.add_argument("--max_patches", type=int, default=0, help="limiter le nombre de positifs lus (0 = tous)")
    ap.add_argument("--pas_journal", type=int, default=200, help="écrire une ligne de progression toutes les N images")
    ap.add_argument("--silencieux", action="store_true", help="ne presque rien afficher dans le terminal")
    args = ap.parse_args()

    global PAS_JOURNAL, AFFICHAGE_TERMINAL
    PAS_JOURNAL = max(1, int(args.pas_journal))
    AFFICHAGE_TERMINAL = 0 if args.silencieux else 1

    construire_gabarit(
        racine_pieton_tm=args.pieton_tm,
        largeur=args.largeur,
        hauteur=args.hauteur,
        ondelette=args.ondelette,
        set_filtre=args.set,
        video_filtre=args.video,
        max_patches=args.max_patches
    )

if __name__ == "__main__":
    main()
    
    
# python gabarit.py --pieton_tm ~/scratch/pieton_tm --pas_journal 200 