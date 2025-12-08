#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Caltech Pedestrian — VBB -> JSON + VIA CSV (un fichier par vidéo)
Sorties (par vidéo) : 
  <out_root>\annotations\<set_name>\<set_name>\<video>\ <video>.json
  <out_root>\via\<set_name>\<set_name>\<video>\       <video>.csv
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

import numpy as np
from scipy.io import loadmat


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_idx(arr, idx, default=-1):
    """Accès robuste pour champs MATLAB: scalaire, vide, (N,), (N,1), (1,N)."""
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
    """Lit un .vbb Caltech et renvoie un dict {nFrame, frames{int->list[objs]}, ...}."""
    vbb = loadmat(str(vbb_path))
    A = vbb['A'][0][0]

    nFrame   = int(A[0][0][0])
    objLists = A[1][0]  # par frame
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
            pid = int(id_[0][0]) - 1  # MATLAB 1-based
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
    """Écrit un JSON pour UNE vidéo (frames: clés string)."""
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
    Exporte un CSV VIA pour UNE vidéo.
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
            fname = f"{set_name}_{video_name}_{fid:06d}.jpg"
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


def main():
    ap = argparse.ArgumentParser(description="Caltech VBB -> JSON + VIA CSV (un fichier par vidéo)")
    ap.add_argument("--ann_dir",  required=True, help="Chemin d’entrée: dossier .vbb ou fichier .vbb")
    ap.add_argument("--set_name", required=True, help="Nom du set (ex: set00)")
    ap.add_argument("--out_root", required=True, help="Dossier racine de sortie")
    ap.add_argument("--via_csv",  action="store_true", help="Générer aussi le CSV pour VIA")
    ap.add_argument("--video",    default="", help="Filtrer une vidéo (ex: V001)")
    args = ap.parse_args()

    ann_input = Path(args.ann_dir)
    if not ann_input.exists():
        print(f"[ERREUR] Chemin introuvable: {ann_input}")
        raise SystemExit(2)

    # Construire la liste de .vbb (dossier ou fichier)
    if ann_input.is_dir():
        vbb_files = sorted(ann_input.glob("*.vbb"))
    else:
        if ann_input.suffix.lower() != ".vbb":
            print(f"[ERREUR] Fichier non .vbb: {ann_input}")
            raise SystemExit(3)
        vbb_files = [ann_input]

    if args.video:
        vbb_files = [p for p in vbb_files if p.stem == args.video]
        if not vbb_files:
            print(f"[ERREUR] {args.video}.vbb introuvable sous: {ann_input.parent if ann_input.is_file() else ann_input}")
            raise SystemExit(4)

    if not vbb_files:
        print(f"[ERREUR] Aucun .vbb trouvé dans: {ann_input}")
        raise SystemExit(3)

    out_root = Path(args.out_root)

    for vbb_path in vbb_files:
        video = vbb_path.stem  # V000, V001, ...
        data  = parse_vbb(vbb_path)

        # Dossiers imbriqués demandés: <out_root>\annotations\setXX\setXX\Vxxx\Vxxx.json
        json_dir = out_root / "annotations" / args.set_name / args.set_name / video
        via_dir  = out_root / "via"         / args.set_name / args.set_name / video
        ensure_dir(json_dir)
        if args.via_csv:
            ensure_dir(via_dir)

        json_path = json_dir / f"{video}.json"
        write_json_per_video(data, json_path)
        print(f"[OK] JSON: {json_path}")

        if args.via_csv:
            csv_path = via_dir / f"{video}.csv"
            write_via_csv_per_video(video, args.set_name, data, csv_path)
            print(f"[OK] VIA:  {csv_path}")

    print("[DONE] Conversion terminée.")


if __name__ == "__main__":
    main()
