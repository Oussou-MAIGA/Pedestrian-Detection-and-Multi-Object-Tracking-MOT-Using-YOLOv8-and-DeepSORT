#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Extraction uniquement du fichier "set00 V000.seq"
# Sauvegarde les frames dans le dossier de sortie donné.

import sys
import struct, json
from pathlib import Path

def detect_format(image_format: int) -> str:
    if image_format in (100, 200): return ".raw"
    if image_format == 101: return ".brgb8"
    if image_format in (102, 201): return ".jpg"
    if image_format == 103: return ".jbrgb"
    if image_format in (1, 2): return ".png"
    return ".bin"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_u32_le(f): return struct.unpack('<I', f.read(4))[0]
def read_f64_le(f): return struct.unpack('<d', f.read(8))[0]

def extract_one_seq(seq_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    with seq_path.open('rb') as f:
        # Sauter le header fixe
        f.seek(28 + 8 + 512)

        # Lecture du header
        keys = ["width","height","imageBitDepth","imageBitDepthReal","imageSizeBytes","imageFormat","numFrames"]
        header = {k: read_u32_le(f) for k in keys}
        f.read(4)  
        header["trueImageSize"] = read_u32_le(f)
        header["fps"] = read_f64_le(f)
        print("[INFO] Header:", header)
        (out_dir / "header.json").write_text(json.dumps(header, indent=2))


        ext = detect_format(header["imageFormat"])
        f.seek(432, 1)

        # Nom court → "V000" pour les images
        stem = "V000"
        n = header["numFrames"]

        for img_id in range(n):
            size = read_u32_le(f)
            img_data = f.read(size)
            img_name = f"{stem}_{img_id:03d}{ext}"
            (out_dir/img_name).write_bytes(img_data)
            f.seek(12, 1)

            if (img_id + 1) % 100 == 0 or img_id == n-1:
                print(f"[INFO] {img_id+1}/{n} frames écrites")

    print(f"[OK] Extraction terminée → {out_dir}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_v000_only.py <V000.seq> <output_dir>")
        sys.exit(1)

    seq_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    extract_one_seq(seq_path, out_dir)

if __name__ == "__main__":
    main()
