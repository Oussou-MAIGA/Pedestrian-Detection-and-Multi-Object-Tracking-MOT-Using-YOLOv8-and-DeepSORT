#!/bin/bash
#SBATCH --job-name=yolo_caltech
#SBATCH --output=/home/ousman/links/scratch/pieton_tm/modeles/yolov8/yolo_caltech_%j.out
#SBATCH --error=/home/ousman/links/scratch/pieton_tm/modeles/yolov8/yolo_caltech_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --partition=compute_full_node
#SBATCH --account=rrg-akhloufi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eom6713@umoncton.ca

# ========= 1) Chargement des modules =========
module purge
module load gcc opencv/4.12.0 python scipy-stack

# ========= 2) Environnement & caches =========
source /home/ousman/links/scratch/venv_pieton/bin/activate

export MPLCONFIGDIR=/scratch/ousman/caltech_prepare/sorties/config/.mpl_cache
export YOLO_CONFIG_DIR=/scratch/ousman/caltech_prepare/sorties/config/.yolo_cache
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

# ========= 3) Variables =========
ROOT=/scratch/ousman/caltech_prepare/sorties
CFG=$ROOT/config
MODEL=/home/ousman/links/scratch/pieton_tm/modeles/yolov8/yolov8s.pt
BEST_PT=/home/ousman/links/scratch/pieton_tm/sorties_detect/yolov8/caltech_person/weights/best.pt
PROJECT_OUT=/home/ousman/links/scratch/pieton_tm/sorties_detect/yolov8

# ========= 4) Génération des listes (train/val/test) =========
echo "[INFO] Génération des listes via $CFG/liste_chemin_image.sh ..."
bash "$CFG/liste_chemin_image.sh"

# ========= 5) Entraînement YOLO (4 époques, 4 GPUs) =========
echo "[INFO] Début entraînement YOLO (4 GPUs, 4 époques)…"
yolo detect train \
  model="$MODEL" \
  data="$CFG/data.yaml" \
  imgsz=640 epochs=4 batch=16 device=0,1,2,3 workers=8 \
  cache=False amp=False verbose=True \
  project="$PROJECT_OUT" \
  name=caltech_person

# ========= 6) Évaluation sur TEST (sets06–10) =========
echo "[INFO] Évaluation sur set06–set10…"
yolo detect val \
  model="$BEST_PT" \
  data="$CFG/data.yaml" \
  split=test imgsz=640 batch=16 device=0,1,2,3 workers=8 \
  cache=False verbose=True \
  project="$PROJECT_OUT" \
  name=caltech_person_test

echo "[DONE] Fin du job."