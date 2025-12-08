#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extraction POSITIFS & NÉGATIFS 64×128 (Papageorgiou-style)
Robuste aux chemins JSON : essaie <src_json>, <src_json>/json, <src_json>/json/json

Lecture :
  FRAMES : <caltech_prepare>/sorties/frames/setXX/VYYY/VYYY_000.jpg + header.json
  JSON   : <src_json>/setXX/VYYY.json  (ou /json/setXX/... ou /json/json/setXX/...)

Écriture :
  <pieton_tm>/dataset_caltech/train/
    positives/setXX/VYYY/pos_*.png + header.json
    negatives/setXX/VYYY/neg_*.png + header.json
"""

import cv2, json, random, argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ---------------- utils FS ----------------
def mkdirs(p: Path): p.mkdir(parents=True, exist_ok=True)
def load_json(p: Path) -> Dict: return json.loads(p.read_text(encoding="utf-8"))
def list_sets(root: Path) -> List[str]: return sorted(d.name for d in root.glob("set*") if d.is_dir())

# ---------------- chemins JSON robustes ----------------
def resolve_json_root(src_json_hint: Path) -> List[Path]:
    """
    Retourne une liste ordonnée de racines JSON candidates à tester :
      1) hint
      2) hint/json
      3) hint/json/json
    On ne garde que celles qui existent.
    """
    cands = [src_json_hint, src_json_hint / "json", src_json_hint / "json" / "json"]
    ok = []
    for c in cands:
        if c.is_dir():
            ok.append(c)
    # déduplique en gardant l'ordre
    seen = set(); roots = []
    for r in ok:
        if r not in seen:
            seen.add(r); roots.append(r)
    return roots

def find_video_json(json_roots: List[Path], set_name: str, video: str) -> Optional[Path]:
    """
    Essaye, pour chaque racine json_root, ces patterns :
      json_root/setXX/VYYY.json
    Retourne le premier qui existe.
    """
    for root in json_roots:
        p = root / set_name / f"{video}.json"
        if p.is_file():
            return p
    return None

# ---------------- chemins images ----------------
def candidate_names(video: str, fid: int) -> List[str]:
    b3, b4 = f"{video}_{fid:03d}", f"{video}_{fid:04d}"
    return [b3+".jpg", b3+".png", b3+".jpeg", b4+".jpg", b4+".png", b4+".jpeg"]

# ---------------- géométrie ----------------
def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

def crop_resize(img, bbox, size=(64,128)):
    x,y,w,h = [int(round(v)) for v in bbox]
    H,W = img.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x));  h = max(1, min(h, H-y))
    crop = img[y:y+h, x:x+w]
    if crop.size == 0: return None
    return cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)

def random_window(W,H,w,h):
    if W<w or H<h: return None
    from random import randint
    return (randint(0, W-w), randint(0, H-h), w, h)

# ---------------- extraction ----------------
def extraire_pos_neg(
    src_frames_root: Path,       # .../sorties/frames
    src_json_hint: Path,         # .../sorties/json   (ou .../sorties/json/json)
    dst_pieton_tm: Path,         # racine pieton_tm
    set_name: str = "auto",
    video_name: str = "",
    largeur: int = 64,
    hauteur: int = 128,
    neg_par_image: int = 3,
    iou_max_neg: float = 0.2,
    niveaux_de_gris: bool = True,
    seed: int = 0
):
    random.seed(seed)

    src_frames_root = src_frames_root.resolve()
    src_json_hint   = src_json_hint.resolve()
    if not src_frames_root.is_dir():
        raise SystemExit(f"[ERREUR] FRAMES introuvables: {src_frames_root}")
    json_roots = resolve_json_root(src_json_hint)
    if not json_roots:
        raise SystemExit(f"[ERREUR] Aucune racine JSON valide trouvée à partir de: {src_json_hint}")

    out_root = dst_pieton_tm.resolve() / "dataset_caltech" / "train"
    out_pos_root = out_root / "positives"
    out_neg_root = out_root / "negatives"
    mkdirs(out_pos_root); mkdirs(out_neg_root)

    # sets pris depuis FRAMES (référence la plus sûre)
    sets = list_sets(src_frames_root) if set_name.lower()=="auto" else [set_name]
    if not sets:
        raise SystemExit(f"[ERREUR] Aucun setXX sous {src_frames_root}")

    tot_pos = tot_neg = tot_frames = tot_gt = 0

    for s in sets:
        frames_set = src_frames_root / s
        if not frames_set.is_dir():
            print(f"[INFO] set sans frames: {s} → ignoré")
            continue
        videos = sorted(d.name for d in frames_set.glob("V*") if d.is_dir())
        if video_name:
            videos = [v for v in videos if v == video_name]
        if not videos:
            print(f"[INFO] aucune vidéo sous {frames_set}")
            continue

        for vid in videos:
            frames_dir = frames_set / vid
            header_path = frames_dir / "header.json"
            if not frames_dir.is_dir():
                print(f"[WARN] frames absents: {s}/{vid}")
                continue

            # résout le JSON d’annotations en testant plusieurs racines
            anno_json = find_video_json(json_roots, s, vid)
            if anno_json is None:
                print(f"[WARN] annotations introuvables pour {s}/{vid} dans {', '.join(map(str,json_roots))} → sauté")
                continue

            try:
                data = load_json(anno_json)
            except Exception as e:
                print(f"[WARN] lecture JSON échouée: {anno_json} ({e}) → sauté")
                continue
            frames = data.get("frames", {})
            
            
            # ---- Comptage des IDs uniques ----
            ids_uniques = set()
            for fid, objs in frames.items():
                for o in objs:
                    if o.get("lbl") == "person" and "id" in o:
                        ids_uniques.add(o["id"])
            nb_ids_uniques = len(ids_uniques)
            print(f"[INFO] {s}/{vid} → {nb_ids_uniques} personnes uniques")

            # header (si présent)
            header_info = {}
            if header_path.is_file():
                try: header_info = load_json(header_path)
                except Exception: pass

            out_pos_dir = out_pos_root / s / vid
            out_neg_dir = out_neg_root / s / vid
            mkdirs(out_pos_dir); mkdirs(out_neg_dir)
            if header_info:
                (out_pos_dir/"header.json").write_text(json.dumps(header_info, indent=2))
                (out_neg_dir/"header.json").write_text(json.dumps(header_info, indent=2))

            npos = nneg = nframes = ngt = 0
            for k in sorted(frames.keys(), key=lambda t: int(t)):
                fid = int(k)
                # image correspondante
                img_path = next((frames_dir / nm for nm in candidate_names(vid, fid)
                                 if (frames_dir / nm).is_file()), None)
                if img_path is None: continue
                img = cv2.imread(str(img_path))
                if img is None: continue
                nframes += 1
                H,W = img.shape[:2]
                work = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if niveaux_de_gris else img

                # GT piétons
                gt = []
                for obj in frames[k]:
                    if obj.get("lbl","person") != "person": continue
                    x,y,w,h = obj.get("pos", [0,0,0,0])
                    if w>0 and h>0:
                        gt.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
                ngt += len(gt)

                # positifs
                for (x,y,w,h) in gt:
                    patch = crop_resize(work, (x,y,w,h), (largeur,hauteur))
                    if patch is None: continue
                    cv2.imwrite(str(out_pos_dir / f"pos_{npos:06d}.png"), patch)
                    npos += 1; tot_pos += 1

                # négatifs (IoU < seuil)
                besoin, essais = neg_par_image, 0
                while besoin > 0 and essais < 200*neg_par_image:
                    essais += 1
                    cand = random_window(W,H,largeur,hauteur)
                    if cand is None: break
                    if max((iou(cand,g) for g in gt), default=0.0) >= iou_max_neg:
                        continue
                    patch = crop_resize(work, cand, (largeur,hauteur))
                    if patch is None: continue
                    cv2.imwrite(str(out_neg_dir / f"neg_{nneg:06d}.png"), patch)
                    nneg += 1; tot_neg += 1; besoin -= 1

            tot_frames += nframes; tot_gt += ngt
            print(f"[OK] {s}/{vid} → +{npos} pos, +{nneg} neg | frames:{nframes} | GT:{ngt}")

    print(f"[FINI] POS={tot_pos}  NEG={tot_neg}  FRAMES={tot_frames}  GT={tot_gt}")
    print(f"[SORTIES] positives: {out_pos_root} | negatives: {out_neg_root}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Caltech → positives/négatives 64×128 (JSON robuste)")
    ap.add_argument("--src_frames",   type=Path, required=True, help="racine frames: .../sorties/frames")
    ap.add_argument("--src_json",     type=Path, required=True, help="racine json vague: .../sorties/json (ou .../sorties/json/json)")
    ap.add_argument("--dst_pieton_tm",type=Path, required=True, help="destination racine pieton_tm")
    ap.add_argument("--set",    default="auto")
    ap.add_argument("--video",  default="")
    ap.add_argument("--largeur", type=int, default=64)
    ap.add_argument("--hauteur", type=int, default=128)
    ap.add_argument("--neg_par_image", type=int, default=3)
    ap.add_argument("--iou_max_neg",   type=float, default=0.2)
    ap.add_argument("--couleur", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    extraire_pos_neg(
        src_frames_root=args.src_frames,
        src_json_hint=args.src_json,
        dst_pieton_tm=args.dst_pieton_tm,
        set_name=args.set,
        video_name=args.video,
        largeur=args.largeur,
        hauteur=args.hauteur,
        neg_par_image=args.neg_par_image,
        iou_max_neg=args.iou_max_neg,
        niveaux_de_gris=not args.couleur,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
    
     
# python patch_and_negatifs.py --src_frames ~/scratch/caltech_prepare/sorties/frames --src_json ~/scratch/caltech_prepare/sorties/json --dst_pieton_tm ~/scratch/pieton_tm --set set00