#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Caltech Pedestrian — VBB -> JSON + VIA CSV (un fichier par vidéo)

Entrée attendue (racine VBB):
  donnees_brute/vbb/            # ou .../vbb/annotations/ (ou .../vbb/annotations/annotations/)
    set00/
      V000.vbb
      V001.vbb
      ...
    set01/
      ...

Sorties (propres) :
  <out_root>/json/<set_name>/V000.json
  <out_root>/via/<set_name>/V000.csv
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
from scipy.io import loadmat


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_idx(arr, idx, default=-1):
    try:
        a = arr
        if a is None:
            return default
        if isinstance(a, (int, float, np.integer, np.floating)):
            return int(a)
        a = np.asarray(a)
        if a.size == 0:
            return default
        if a.ndim == 0:
            return int(a)
        if a.ndim == 1:
            if idx < 0 or idx >= a.shape[0]:
                return default
            return int(np.asarray(a[idx]).flat[0])
        if a.ndim == 2:
            if a.shape[0] >= a.shape[1]:
                if idx < 0 or idx >= a.shape[0]:
                    return default
                val = a[idx, ...]
            else:
                if idx < 0 or idx >= a.shape[1]:
                    return default
                val = a[..., idx]
            return int(np.asarray(val).flat[0])
        return int(np.asarray(a).flat[0])
    except Exception:
        return default


def parse_vbb(vbb_path: Path) -> Dict[str, Any]:
    vbb = loadmat(str(vbb_path))
    A = vbb['A'][0][0]

    nFrame   = int(A[0][0][0])
    objLists = A[1][0]
    maxObj   = int(A[2][0][0])
    objInit  = A[3][0]
    objLbl   = [str(x[0]) for x in A[4][0]]
    objStr   = A[5][0]
    objEnd   = A[6][0]
    objHide  = A[7][0]
    altered  = int(A[8][0][0])
    log      = A[9][0] if A[9].size > 0 else np.array([])
    logLen   = int(A[10][0][0])

    data = {
        'nFrame': nFrame,
        'maxObj': maxObj,
        'altered': altered,
        'logLen': logLen,
        'log': log.tolist() if isinstance(log, np.ndarray) else [],
        'frames': defaultdict(list)
    }

    for fid, obj in enumerate(objLists):
        if len(obj) == 0:
            continue
        ids   = obj['id'][0]
        poss  = obj['pos'][0]
        occls = obj['occl'][0]
        locks = obj['lock'][0]
        posvs = obj['posv'][0]

        for id_, pos_, occl_, lock_, posv_ in zip(ids, poss, occls, locks, posvs):
            pid = int(id_[0][0]) - 1
            pos  = [float(p) for p in pos_[0]] if len(pos_) > 0 else [0.0, 0.0, 0.0, 0.0]
            occl = int(occl_[0][0]) if len(occl_) > 0 else 0
            lock = int(lock_[0][0]) if len(lock_) > 0 else 0
            posv = [float(p) for p in posv_[0]] if len(posv_) > 0 else [0.0, 0.0, 0.0, 0.0]

            data['frames'][fid].append({
                'id':   pid,
                'pos':  pos,  # [x,y,w,h]
                'occl': occl,
                'lock': lock,
                'posv': posv,
                'lbl':  str(objLbl[pid]) if 0 <= pid < len(objLbl) else 'unknown',
                'str':  safe_idx(objStr,  pid, -1),
                'end':  safe_idx(objEnd,  pid, -1),
                'hide': safe_idx(objHide, pid,  0),
                'init': safe_idx(objInit, pid,  0),
            })
    return data


def write_json_per_video(data: Dict[str, Any], out_path: Path):
    ensure_dir(out_path.parent)
    payload = {
        "nFrame": data["nFrame"],
        "maxObj": data["maxObj"],
        "altered": data["altered"],
        "logLen": data["logLen"],
        "log": data["log"],
        "frames": {str(fid): objs for fid, objs in data["frames"].items()}
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_via_csv_per_video(video_name: str, set_name: str, data: Dict[str, Any], out_csv: Path):
    """
    VIA CSV (1 fichier par vidéo).
    Hypothèse d’images: <set_name>_<video_name>_%06d.jpg (adapte si besoin).
    """
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename","file_size","file_attributes",
            "region_count","region_id",
            "region_shape_attributes","region_attributes"
        ])
        writer.writeheader()

        for fid, objs in data["frames"].items():
            
            fid_int = int(fid)
            fname = f"{set_name}_{video_name}_{fid_int:06d}.jpg"
            for rid, o in enumerate(objs):
                x, y, w, h = o.get("pos", [0,0,0,0])
                shape = json.dumps({"name":"rect","x":x,"y":y,"width":w,"height":h})
                attrs = json.dumps({"class": o.get("lbl","person")})
                writer.writerow({
                    "filename": fname,
                    "file_size": "",
                    "file_attributes": "{}",
                    "region_count": "",
                    "region_id": rid,
                    "region_shape_attributes": shape,
                    "region_attributes": attrs
                })


def resolve_set_dir(ann_root: Path, set_name: str) -> Path:
    """
    Résout le dossier setXX, en tolérant 'annotations' répété:
      ann_root/setXX
      ann_root/annotations/setXX
      ann_root/annotations/annotations/setXX
    """
    candidates: List[Path] = [
        ann_root / set_name,
        ann_root / "annotations" / set_name,
        ann_root / "annotations" / "annotations" / set_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"set introuvable: {set_name} sous {ann_root}")


def main():
    ap = argparse.ArgumentParser(description="Caltech VBB -> JSON + VIA CSV (un fichier par vidéo)")
    ap.add_argument("--ann_root", required=True, help="Racine des VBB (dossier contenant setXX/ ou .../annotations/setXX/)")
    ap.add_argument("--set_name", required=True, help="Nom du set (ex: set00)")
    ap.add_argument("--out_root", required=True, help="Dossier racine de sortie")
    ap.add_argument("--via_csv",  action="store_true", help="Générer aussi le CSV pour VIA")
    ap.add_argument("--video",    default="", help="Filtrer une vidéo (ex: V001)")
    args = ap.parse_args()

    ann_root = Path(args.ann_root).expanduser()
    if not ann_root.exists():
        print(f"[ERREUR] Racine introuvable: {ann_root}")
        raise SystemExit(2)

    set_dir = resolve_set_dir(ann_root, args.set_name)
    vbb_files = sorted(set_dir.glob("V*.vbb"))
    if args.video:
        vbb_files = [p for p in vbb_files if p.stem == args.video]
    if not vbb_files:
        print(f"[ERREUR] Aucun .vbb trouvé dans: {set_dir}")
        raise SystemExit(3)

    out_root = Path(args.out_root).expanduser()
    out_json = out_root / "json" / args.set_name
    out_via  = out_root / "via"  / args.set_name
    ensure_dir(out_json)
    if args.via_csv:
        ensure_dir(out_via)

    for vbb_path in vbb_files:
        video = vbb_path.stem  # V000
        data  = parse_vbb(vbb_path)

        json_path = out_json / f"{video}.json"
        write_json_per_video(data, json_path)
        print(f"[OK] JSON: {json_path}")

        if args.via_csv:
            csv_path = out_via / f"{video}.csv"
            write_via_csv_per_video(video, args.set_name, data, csv_path)
            print(f"[OK] VIA:  {csv_path}")

    print("[DONE] Conversion terminée.")


if __name__ == "__main__":
    main()
    
    
    
''' expmple de lancement
python convertir_vbb.py --ann_root ~/links/scratch/caltech_prepare/donnees_brute/vbb/annotations/annotations --set_name set10 --out_root ~/links/scratch/caltech_prepare/sorties/json  --via_csv
python convertir_vbb.py --ann_root ~/links/scratch/caltech_prepare/donnees_brute/vbb/annotations/annotations --set_name set10 --out_root ~/links/scratch/caltech_prepare/sorties/json  --via_csv --video V000
'''