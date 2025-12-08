#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patches_qualite.py
Extraction de POSITIFS & NÉGATIFS de bonne qualité (Caltech-like) + rapports.

Obligatoire : --set setXX
- Si --video est omis → traite toutes les vidéos V*** sous frames/<set>/.
- Si --video VYYY est fourni → traite uniquement cette vidéo.

Lecture :
  frames/<set>/VYYY/VYYY_000.jpg (+ header.json)
  annotations : <src_json>/<set>/VYYY.json   (le script résout aussi .../json et .../json/json)

Sorties (dans --dst_root) :
  dataset_caltech/train/
    ├── positifs_qualite/<set>/<vid>/pos_*.png (+ header.json, rapport_qualite.txt)
    ├── negatifs_qualite/<set>/<vid>/neg_*.png (+ header.json, rapport_qualite.txt)
    └── rapports_qualite/<set>.json      (récap global du set)
"""

import cv2, json, random, argparse
from pathlib import Path
import numpy as np
from datetime import datetime

# ----------------- utilitaires -----------------
def mkdirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def charger_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def est_flou(img, seuil_var_lapl=80.0):
    """True si l'image est floue (variance du Laplacien faible)."""
    return cv2.Laplacian(img, cv2.CV_64F).var() < float(seuil_var_lapl)

def faible_contraste(img, seuil_std=25.0):
    """True si contraste très faible."""
    return img.std() < float(seuil_std)

def recadrer_redimensionner(img, bbox, taille=(64,128)):
    x,y,w,h = [int(round(v)) for v in bbox]
    H,W = img.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x));  h = max(1, min(h, H-y))
    crop = img[y:y+h, x:x+w]
    if crop.size == 0: return None
    return cv2.resize(crop, taille, interpolation=cv2.INTER_LINEAR)

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def fenetre_aleatoire(W,H,w,h):
    if W<w or H<h: return None
    from random import randint
    return (randint(0, W-w), randint(0, H-h), w, h)

def noms_images(video: str, fid: int):
    b3 = f"{video}_{fid:03d}"
    b4 = f"{video}_{fid:04d}"
    return [b3+".jpg", b3+".png", b3+".jpeg", b4+".jpg", b4+".png", b4+".jpeg"]

def resoudre_racines_json(racine_hint: Path):
    cands = [racine_hint, racine_hint/"json", racine_hint/"json"/"json"]
    return [c for c in cands if c.is_dir()]

def trouver_json_video(racines_json, set_name, video):
    for r in racines_json:
        p = r/set_name/f"{video}.json"
        if p.is_file(): return p
    return None

def ecrire_rapport_video(fichier: Path, stats: dict, seuils: dict):
    contenu = []
    contenu.append(f"date: {datetime.now().isoformat()}")
    contenu.append(f"set: {stats['set']}  video: {stats['video']}")
    contenu.append(f"taille_patch: {stats['taille'][0]}x{stats['taille'][1]}")
    contenu.append(f"seuil_flou(varLapl): {seuils['flou']}  seuil_contraste(std): {seuils['contraste']}")
    contenu.append(f"frames_lues: {stats['frames']}")
    contenu.append(f"gt_total: {stats['gt_total']}")
    contenu.append(f"positifs_gardes: {stats['pos_keep']}  rejetes_flou: {stats['pos_rej_blur']}  rejetes_contraste: {stats['pos_rej_contrast']}")
    contenu.append(f"negatifs_gardes: {stats['neg_keep']}  rejetes_flou: {stats['neg_rej_blur']}  rejetes_contraste: {stats['neg_rej_contrast']}")
    contenu.append(f"neg_tentatives: {stats['neg_trials']}")
    fichier.write_text("\n".join(contenu), encoding="utf-8")

# ----------------- extraction -----------------
def extraire_patches_qualite(
    src_frames_root: Path,
    src_json_root: Path,
    dst_root: Path,
    set_name: str,
    video_name: str = "",
    largeur: int = 64,
    hauteur: int = 128,
    neg_par_image: int = 3,
    iou_max_neg: float = 0.2,
    seuil_flou: float = 80.0,
    seuil_contraste: float = 25.0,
    seed: int = 0
):
    random.seed(seed)

    frames_set = src_frames_root / set_name
    if not frames_set.is_dir():
        raise SystemExit(f"[ERREUR] set introuvable sous frames: {frames_set}")

    videos = sorted(d.name for d in frames_set.glob("V*") if d.is_dir())
    if video_name:
        videos = [v for v in videos if v == video_name]
        if not videos:
            raise SystemExit(f"[ERREUR] vidéo {video_name} introuvable sous {frames_set}")

    racines_json = resoudre_racines_json(src_json_root)
    if not racines_json:
        raise SystemExit(f"[ERREUR] racine JSON invalide: {src_json_root}")

    base_out = dst_root 
    out_pos_base = base_out / "positifs_qualite"
    out_neg_base = base_out / "negatifs_qualite"
    out_reports  = base_out / "rapports_qualite"
    mkdirs(out_pos_base); mkdirs(out_neg_base); mkdirs(out_reports)

    # --- Récap JSON global du set ---
    recap_json = out_reports / f"{set_name}.json"
    if recap_json.exists():
        recap_data = json.loads(recap_json.read_text(encoding="utf-8"))
    else:
        recap_data = {"set": set_name, "created": datetime.now().isoformat(), "videos": []}

    total_pos = total_neg = 0

    for vid in videos:
        frames_dir = frames_set / vid
        if not frames_dir.is_dir():
            print(f"[WARN] frames absents: {frames_dir} → sauté")
            continue

        anno_json = trouver_json_video(racines_json, set_name, vid)
        if anno_json is None:
            print(f"[WARN] annotations introuvables pour {set_name}/{vid} → sauté")
            continue
        data = charger_json(anno_json)
        frames = data.get("frames", {})

        out_pos = out_pos_base / set_name / vid
        out_neg = out_neg_base / set_name / vid
        mkdirs(out_pos); mkdirs(out_neg)

        hdr = frames_dir / "header.json"
        if hdr.is_file():
            (out_pos/"header.json").write_text(hdr.read_text())
            (out_neg/"header.json").write_text(hdr.read_text())

        # stats vidéo
        stats = {
            "set": set_name, "video": vid,
            "frames": 0, "gt_total": 0,
            "pos_keep": 0, "pos_rej_blur": 0, "pos_rej_contrast": 0,
            "neg_keep": 0, "neg_rej_blur": 0, "neg_rej_contrast": 0,
            "neg_trials": 0,
            "taille": [largeur, hauteur],
            "seuil_flou": float(seuil_flou),
            "seuil_contraste": float(seuil_contraste)
        }

        for k in sorted(frames.keys(), key=lambda t: int(t)):
            fid = int(k)
            # image correspondante
            img_path = None
            for nm in noms_images(vid, fid):
                p = frames_dir / nm
                if p.is_file():
                    img_path = p; break
            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H,W = gray.shape
            stats["frames"] += 1

            # GT
            objs = [o for o in frames[str(fid)] if o.get("lbl")=="person"]
            gts = []
            for o in objs:
                x,y,w,h = o.get("pos",[0,0,0,0])
                if w>0 and h>0: gts.append((int(round(x)),int(round(y)),int(round(w)),int(round(h))))
            stats["gt_total"] += len(gts)

            # POSITIFS
            for (x,y,w,h) in gts:
                patch = recadrer_redimensionner(gray, (x,y,w,h), (largeur,hauteur))
                if patch is None:
                    continue
                if est_flou(patch, seuil_var_lapl=seuil_flou):
                    stats["pos_rej_blur"] += 1; continue
                if faible_contraste(patch, seuil_std=seuil_contraste):
                    stats["pos_rej_contrast"] += 1; continue
                cv2.imwrite(str(out_pos / f"pos_{stats['pos_keep']:06d}.png"), patch)
                stats["pos_keep"] += 1; total_pos += 1

            # NÉGATIFS
            besoin, essais = neg_par_image, 0
            while besoin>0 and essais<150*neg_par_image:
                essais += 1; stats["neg_trials"] += 1
                cand = fenetre_aleatoire(W,H,largeur,hauteur)
                if cand is None: break
                if max((iou(cand, g) for g in gts), default=0.0) >= iou_max_neg:
                    continue
                patch = recadrer_redimensionner(gray, cand, (largeur,hauteur))
                if patch is None:
                    continue
                if est_flou(patch, seuil_var_lapl=seuil_flou):
                    stats["neg_rej_blur"] += 1; continue
                if faible_contraste(patch, seuil_std=seuil_contraste):
                    stats["neg_rej_contrast"] += 1; continue
                cv2.imwrite(str(out_neg / f"neg_{stats['neg_keep']:06d}.png"), patch)
                stats["neg_keep"] += 1; total_neg += 1; besoin -= 1

        # rapport texte par vidéo
        seuils = {"flou": seuil_flou, "contraste": seuil_contraste}
        ecrire_rapport_video(out_pos/"rapport_qualite.txt", stats, seuils)
        ecrire_rapport_video(out_neg/"rapport_qualite.txt", stats, seuils)

        # ajoute la vidéo au récap JSON
        recap_data["videos"].append(stats)

        print(f"[OK] {set_name}/{vid} → pos_keep={stats['pos_keep']}  neg_keep={stats['neg_keep']} "
              f"(rej_pos: blur={stats['pos_rej_blur']}, ctr={stats['pos_rej_contrast']}; "
              f"rej_neg: blur={stats['neg_rej_blur']}, ctr={stats['neg_rej_contrast']})")

    # --- Sauvegarde du récap JSON du set ---
    recap_json.write_text(json.dumps(recap_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FINI {set_name}] total_pos={total_pos}  total_neg={total_neg}")
    print(f"[OK] Récap JSON → {recap_json}")

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Extraction POS/NEG de bonne qualité (Caltech) + rapports")
    ap.add_argument("--src_frames", type=Path, required=True, help="racine frames: .../sorties/frames")
    ap.add_argument("--src_json",   type=Path, required=True, help="racine json: .../sorties/json (ou .../sorties/json/json)")
    ap.add_argument("--dst_root",   type=Path, required=True, help="racine de sortie (pieton_tm)")
    ap.add_argument("--set",        required=True, help="ex: set00 [OBLIGATOIRE]")
    ap.add_argument("--video",      default="",    help="ex: V000 (optionnel). Si omis → toutes les vidéos du set")
    ap.add_argument("--largeur",    type=int, default=64)
    ap.add_argument("--hauteur",    type=int, default=128)
    ap.add_argument("--neg_par_image", type=int, default=3)
    ap.add_argument("--iou_max_neg",   type=float, default=0.2)
    ap.add_argument("--seuil_flou",    type=float, default=80.0)
    ap.add_argument("--seuil_contraste", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    extraire_patches_qualite(
        src_frames_root=args.src_frames,
        src_json_root=args.src_json,
        dst_root=args.dst_root,
        set_name=args.set,
        video_name=args.video,
        largeur=args.largeur,
        hauteur=args.hauteur,
        neg_par_image=args.neg_par_image,
        iou_max_neg=args.iou_max_neg,
        seuil_flou=args.seuil_flou,
        seuil_contraste=args.seuil_contraste,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
    
# python patches_qualite.py --src_frames ~/links/scratch/caltech_prepare/sorties/frames --src_json ~/links/scratch/caltech_prepare/sorties/json --dst_root ~/links/scratch/pieton_tm/dataset_caltech/train --set set00 --video V000