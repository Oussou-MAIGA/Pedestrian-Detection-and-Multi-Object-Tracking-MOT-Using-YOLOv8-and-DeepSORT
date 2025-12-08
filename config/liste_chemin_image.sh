#!/usr/bin/env bash
set -euo pipefail

# Racine unique de MON dataset (frames + labels)
ROOT=/scratch/ousman/caltech_prepare/sorties
CFG="$ROOT/config"

# ----- Chemins explicites  -----
TRAIN_LABEL_DIRS="$ROOT/labels/set00 $ROOT/labels/set01 $ROOT/labels/set02 $ROOT/labels/set03 $ROOT/labels/set04"
VAL_LABEL_DIRS="$ROOT/labels/set05"
TEST_LABEL_DIRS="$ROOT/labels/set06 $ROOT/labels/set07 $ROOT/labels/set08 $ROOT/labels/set09 $ROOT/labels/set10"

TRAIN_FRAME_DIRS="$ROOT/frames/set00 $ROOT/frames/set01 $ROOT/frames/set02 $ROOT/frames/set03 $ROOT/frames/set04"
VAL_FRAME_DIRS="$ROOT/frames/set05"
TEST_FRAME_DIRS="$ROOT/frames/set06 $ROOT/frames/set07 $ROOT/frames/set08 $ROOT/frames/set09 $ROOT/frames/set10"

mkdir -p "$CFG"

# YOLO infère les labels en remplaçant /images/ -> /labels/
ln -sfn "$ROOT/frames" "$ROOT/images"

# --- helpers ---------------------------------------------------------------
mk_pos_list() {  # labels non vides -> chemins IMAGES
  # usage: mk_pos_list out.txt "<label_dir1> <label_dir2> ..."
  out=$1; dirs=$2
  # shellcheck disable=SC2086
  find $dirs -type f -name '*.txt' ! -empty \
    | sed 's#/labels/#/images/#; s#\.txt$#.jpg#' \
    | sort > "$out"
}

# (optionnel, debug) négatifs (vides/absents) depuis frames
mk_neg_list_optional() {
  # usage: mk_neg_list_optional out.txt "<frame_dir1> <frame_dir2> ..."
  out=$1; framedirs=$2
  tmp=$(mktemp)

  find $framedirs -type f -name '*.jpg' \
  | awk '
      {
        img=$0; lbl=img;
        sub("/frames/","/labels/",lbl); sub(/\.jpg$/,".txt",lbl);
        cmd="[ -s \"" lbl "\" ]";   # existe et non vide ? alors POS -> on garde pas
        code=system(cmd);
        if (code!=0) print img;     # sinon NEG (vide ou absent)
      }' \
  | sort > "$tmp"
  mv "$tmp" "$out"
}
# ---------------------------------------------------------------------------

echo "[TRAIN+] génération des positifs…"
mk_pos_list "$CFG/train_pos_images.txt" "$TRAIN_LABEL_DIRS"

echo "[VAL+] génération des positifs…"
mk_pos_list "$CFG/val_pos_images.txt" "$VAL_LABEL_DIRS"

echo "[TEST+] génération des positifs…"
mk_pos_list "$CFG/test_pos_images.txt" "$TEST_LABEL_DIRS"

# (facultatif) listes négatives pour contrôle (non utilisées par MON yaml)
mk_neg_list_optional "$CFG/train_neg_images.txt" "$TRAIN_FRAME_DIRS"
mk_neg_list_optional "$CFG/val_neg_images.txt"   "$VAL_FRAME_DIRS"
mk_neg_list_optional "$CFG/test_neg_images.txt"  "$TEST_FRAME_DIRS"

# Résumé rapide
echo "-------------------- RÉSUMÉ --------------------"
printf "train_pos_images.txt : %s\n" "$(wc -l < "$CFG/train_pos_images.txt")"
printf "val_pos_images.txt   : %s\n" "$(wc -l < "$CFG/val_pos_images.txt")"
printf "test_pos_images.txt  : %s\n" "$(wc -l < "$CFG/test_pos_images.txt")"
printf "(debug) train_neg    : %s\n" "$(wc -l < "$CFG/train_neg_images.txt")"
printf "(debug) val_neg      : %s\n" "$(wc -l < "$CFG/val_neg_images.txt")"
printf "(debug) test_neg     : %s\n" "$(wc -l < "$CFG/test_neg_images.txt")"

# Sanity d’un exemple POS
p=$(head -n1 "$CFG/train_pos_images.txt" || true)
if [ -n "${p:-}" ]; then
  lbl=${p/\/images\//\/labels\/}; lbl=${lbl%.jpg}.txt
  echo "exemple POS img: $p"
  echo "exemple POS lbl: $lbl"
  head -n1 "$lbl" || true
fi

echo "[OK] Listes alignées avec le data.yaml."
