---

# 🛠 Technologies utilisées

<p align="left">

  <!-- Python -->
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" height="28"/>

  <!-- Ultralytics -->
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-brightgreen?logo=ultralytics" height="28"/>

  <!-- OpenCV -->
  <img src="https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv" height="28"/>

  <!-- Scikit-learn -->
  <img src="https://img.shields.io/badge/Scikit--Learn-SVM-orange?logo=scikitlearn" height="28"/>

  <!-- PyWavelets -->
  <img src="https://img.shields.io/badge/PyWavelets-Filtering-yellow" height="28"/>

  <!-- DeepSORT -->
  <img src="https://img.shields.io/badge/DeepSORT-ReID%20MobileNet-purple" height="28"/>

  <!-- ByteTrack -->
  <img src="https://img.shields.io/badge/ByteTrack-MOT-blueviolet" height="28"/>

  <!-- SLURM -->
  <img src="https://img.shields.io/badge/SLURM-HPC%20Cluster-green?logo=linux" height="28"/>

  <!-- GPU -->
  <img src="https://img.shields.io/badge/NVIDIA-H100%20GPU-76B900?logo=nvidia&logoColor=white" height="28"/>

</p>

---

# Détection et Suivi de Piétons  
**Haar/SVM · HOG/SVM · YOLOv8s · DeepSORT (ReID MobileNet) · ByteTrack**

Projet du cours — Université de Moncton  
Auteur : **Ousmane Maiga**  
Superviseur : **Pr. Moulay Akhloufi – PRIME Lab**

---

# 1. Description du projet

Ce projet compare trois approches de **détection de piétons** :

- Haar + SVM  
- HOG + SVM  
- YOLOv8s (meilleur modèle)

et deux méthodes de **suivi multi-objets** :

- DeepSORT (avec ReID MobileNet)  
- ByteTrack (implémentation Ultralytics)

Objectifs :

- analyser pourquoi les détecteurs classiques échouent en scène réelle  
- étudier la généralisation cross-dataset (**Caltech → INRIA**)  
- mesurer l’impact de la qualité des détections sur le tracking  
- produire des résultats visuels et deux vidéos finales de suivi

---

# 2. Structure du projet

```text
projet_detection_suivi_pietons/
│
├── README.md
├── train_yolo.slurm
│
├── images/                 # résultats de détection pour le rapport / README
│   ├── haar_caltech_1.png
│   ├── haar_caltech_2.png
│   ├── haar_inria_1.png
│   ├── haar_inria_2.png
│   ├── hog_inria_1.png
│   ├── hog_inria_2.png
│   ├── yolo_caltech_inria_1.jpg
│   └── yolo_caltech_inria_2.jpg
│
├── videos/                 # résultats de suivi (DeepSORT / ByteTrack)
│   ├── DeepSort.mp4
│   └── ByteTrack.mp4
│
├── modeles/
│   └── caltech_person/
│       └── weights/
│           └── best.pt    # meilleur modèle YOLOv8s (entraîné sur Caltech)
│
├── datasets/               # à remplir via les liens officiels (Section 3)
│   ├── Caltech/
│   ├── INRIA/
│   └── KITTI/
│
├── scripts/
│   ├── feature_haar_inria.py
│   ├── features_hog_inria.py
│   ├── patch_and_negatifs_inria.py
│   ├── entrainement_svm_inria.py
│   ├── entrainement_svm_hog_inria.py
│   ├── detect_inria_svm.py
│   ├── detect_inria_hog_svm.py
│   ├── track_ReID_deepsort.py
│   ├── eval_MOT.py
│   ├── convert_Pred_to_MOT.py
│   ├── convert_kitti_GT_to_MOT.py
│   ├── extract_images.py
│   ├── extract_annotations.py
│   ├── convertir_vbb.py
│   ├── video_to_frames.py
│   └── images_to_videos.py
│
└── config/
    ├── data_caltech.yaml
    ├── data_inria.yaml
    └── liste_chemin_image.sh


```

# 3. Datasets (liens officiels)
Les datasets sont trop volumineux pour être versionnés.
Ils doivent être téléchargés depuis les sites officiels puis placés dans datasets/.

🔹 Caltech Pedestrian
Site : https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

Dossier cible : datasets/Caltech/

Les scripts convertir_vbb.py, extract_images.py, extract_annotations.py
permettent de convertir .seq + .vbb → images + labels YOLO.

🔹 INRIA Person
Repo : https://github.com/olt/inria-object-detection

Dossier cible : datasets/INRIA/

🔹 KITTI Tracking 
Site : https://www.cvlibs.net/datasets/kitti/eval_tracking.php

Dossier cible : datasets/KITTI/

Une fois les archives KITTI extraites, vous obtenez la structure officielle, par exemple :

datasets/KITTI/
 └── tracking/
     └── training/
         └── image_02/
             ├── 0000/
             ├── 0001/
             ├── 0012/
             ├── 0019/
             └── ...

L’idée est simplement de placer les images de tracking dans datasets/KITTI/...
en respectant l’organisation native de KITTI.

# 4. Résultats de détection

Haar + SVM (Caltech / INRIA)
<p align="center"> <img src="images/haar_caltech_1.png" width="260"/> <img src="images/haar_caltech_2.png" width="260"/> </p> <p align="center"> <img src="images/haar_inria_1.png" width="260"/> <img src="images/haar_inria_2.png" width="260"/> </p>
HOG + SVM (INRIA)
<p align="center"> <img src="images/hog_inria_1.png" width="260"/> <img src="images/hog_inria_2.png" width="260"/> </p>
YOLOv8s (modèle entraîné sur Caltech, testé sur INRIA)
<p align="center"> <img src="images/yolo_caltech_inria_1.jpg" width="260"/> <img src="images/yolo_caltech_inria_2.jpg" width="260"/> </p>

# 5. Résultats de suivi
Les vidéos finales de suivi sont dans :

videos/DeepSort.mp4

videos/ByteTrack.mp4

DeepSORT
Voir la vidéo DeepSORT

ByteTrack
Voir la vidéo ByteTrack

# 6. Environnement logiciel (Cluster Trilium)
Sur le cluster Trilium, avant d’exécuter l’entraînement ou les évaluations YOLO,
les modules et bibliothèques suivants sont chargés / installés :

module load python/3.11.5
module load gcc opencv/4.12.0 python script-stick

# activation de l'environnement virtuel (exemple)
source /chemin/vers/mon_env/bin/activate

# installation des dépendances
pip install --no-index \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  pywavelets scikit-learn ultralytics

Ces commandes sont exécutées avant :

sbatch train_yolo.slurm

yolo detect val ...

yolo detect predict ...

yolo track ...

# 7. Modèle YOLOv8s (base + fine-tuning)
## 7.1 Modèle de base (pré-entraîné COCO)
Fichier : yolov8s.pt

Téléchargement officiel :
https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt

Ce modèle est utilisé dans train_yolo.slurm comme point de départ :

MODEL="yolov8s.pt"

## 7.2 Modèle final (fine-tuné sur Caltech)
Le fine-tuning sur Caltech produit le meilleur modèle du projet (celui utilisé dans l’article) :

modeles/caltech_person/weights/best.pt
Ce modèle :

est entraîné sur Caltech (train/val),

est évalué automatiquement sur Caltech (test) dans le script SLURM (yolo val split=test),

est ensuite réutilisé pour l’évaluation Caltech → INRIA (cross-dataset),

sert de modèle unique pour tous les tests et pour les deux trackers (DeepSORT et ByteTrack).

# 8. Entraînement YOLOv8s sur Caltech
L’entraînement se fait via le script SLURM :

sbatch train_yolo.slurm
Dans ce script :

model=$MODEL pointe vers yolov8s.pt (modèle de base COCO),

data=config/data_caltech.yaml décrit les chemins du dataset Caltech converti en format YOLO,

les paramètres par défaut (batch, epochs, etc.) sont ajustés pour le cluster.

À la fin de l’entraînement, Ultralytics valide automatiquement sur le split test de Caltech :

yolo detect val \
  model=modeles/caltech_person/weights/best.pt \
  data=config/data_caltech.yaml \
  split=test

Ce yolo val fournit les performances officielles Caltech → Caltech
utilisées dans le rapport (mAP@50, F1, etc.).

Temps d’exécution observé sur Trilium :

~ 2 h 05 min 41 s sur 4 GPUs (H100).

# 9. Évaluation YOLOv8s (cross-dataset Caltech → INRIA)
Après l’entraînement sur Caltech, on réutilise le même modèle :

yolo detect val \
  model=modeles/caltech_person/weights/best.pt \
  data=config/data_inria.yaml \
  split=test

Résultat principal (Caltech → INRIA) :

mAP@50 ≈ 0.689

F1 et PR détaillés dans l’article (courbes PR/F1 + matrice de confusion).

Dans le rapport, c’est ce cas Caltech → INRIA qui est considéré comme
meilleur scénario global (modèle entraîné sur un dataset plus difficile et testé sur un plus simple).

# 10. Suivi multi-objets
## 10.1 DeepSORT (ReID MobileNet)

DeepSORT n’est pas intégré directement dans Ultralytics :
on utilise le script Python track_ReID_deepsort.py, qui prend en entrée :

les images KITTI pour une séquence (ex. 0019),

les détections YOLOv8s au format e (.txt) générées par Ultralytics,

un dossier de sortie pour les frames annotées et les labels avec ID.

### 10.1.1 Générer les détections YOLO sur KITTI

yolo detect predict \
  model=modeles/caltech_person/weights/best.pt \
  source=datasets/KITTI/tracking/training/image_02/0019 \
  imgsz=1408 \
  conf=0.60 \
  save=True \
  save_txt=True \
  project=runs/detect \
  name=kitti_0019_yolo

Cela produit une structure de ce type :

runs/detect/kitti_0019_yolo/
 ├── 000000.png
 ├── 000001.png
 ├── ...
 └── labels/
      ├── 000000.txt    # cls cx cy w h conf
      ├── 000001.txt
      └── ...
### 10.1.2 Lancer DeepSORT


python scripts/track_ReID_deepsort.py \
  --img_dir  datasets/KITTI/tracking/training/image_02/0019 \
  --dets_dir runs/detect/kitti_0019_yolo/labels \
  --out_dir  runs/tracking/deepsort_0019 \
  --embedder mobilenet \
  --max_age 10 \
  --n_init 3 \
  --max_cosine_distance 0.4

Paramètres principaux :

--img_dir : images KITTI d’une séquence (ex. 0019)

--dets_dir : fichiers .txt YOLO générés par yolo detect predict

--out_dir : dossier de sortie des résultats DeepSORT

--embedder : modèle ReID utilisé (mobilenet)

--max_age : durée de vie d’une piste sans détection

--n_init : nombre de frames nécessaires pour valider une piste

--max_cosine_distance : seuil d’acceptation pour la similarité d’apparence

Résultats :

runs/tracking/deepsort_0019/
 ├── frames/
 │    ├── 000000.png      # image annotée (bbox + ID)
 │    ├── 000001.png
 │    └── ...
 └── labels/
      ├── 000000.txt      # cls cx cy w h track_id
      ├── 000001.txt
      └── ...
Les vidéos finales visibles dans videos/DeepSort.mp4 sont construites
à partir de ces frames via images_to_videos.py.

## 10.2 ByteTrack (Ultralytics)
ByteTrack est directement intégré dans Ultralytics via yolo track.

Commande d’exemple (séquence KITTI 0019)

yolo track \
  model="modeles/caltech_person/weights/best.pt" \
  source="datasets/KITTI/tracking/training/image_02/0019" \
  imgsz=1408 \
  conf=0.60 \
  tracker="bytetrack.yaml" \
  save=True \
  save_txt=True \
  save_json=True \
  project="runs/kitti_eval" \
  name="bytetrack_0019"

model= : modèle YOLOv8s fine-tuné sur Caltech

source= : dossier d’images KITTI pour une séquence

tracker="bytetrack.yaml" : active ByteTrack

save=True : enregistre la vidéo annotée (.mp4)

save_txt=True : enregistre les labels avec track_id

save_json=True : exporte les résultats en JSON (format MOT-compatible)

Sorties typiques :

runs/kitti_eval/bytetrack_0019/
 ├── bytetrack_0019.mp4        # vidéo annotée
 ├── labels/
 │    ├── 000000.txt           # cls cx cy w h track_id
 │    ├── 000001.txt
 │    └── ...
 └── predictions.json          # résultats pour évaluation MOT

Ces fichiers peuvent ensuite être convertis et évalués avec :

scripts/convert_Pred_to_MOT.py

scripts/eval_MOT.py

pour obtenir les métriques IDF1, MOTA, etc., comme dans l’article.

# 11. Reproductibilité (résumé)
Charger l’environnement Trilium (Section 6)

Télécharger et placer les datasets (Section 3)

Convertir Caltech en images + YOLO (scripts convertir_vbb.py, extract_images.py, extract_annotations.py)

Générer les splits :

 config/liste_chemin_image.sh

Entraîner YOLOv8s sur Caltech :

sbatch train_yolo.slurm

→ modèle : modeles/caltech_person/weights/best.pt
→ validation automatique Caltech → Caltech (yolo val split=test)

Évaluer Caltech → INRIA :

yolo detect val \
  model=modeles/caltech_person/weights/best.pt \
  data=config/data_inria.yaml \
  split=test

Générer les détections KITTI (pour le tracking) avec yolo detect predict.

Lancer DeepSORT avec track_ReID_deepsort.py.

Lancer ByteTrack avec yolo track ... tracker="bytetrack.yaml".

# 12. Modèle final du projet
Le modèle unique utilisé dans tous les résultats de l’article est :

modeles/caltech_person/weights/best.pt
entraîné sur Caltech

évalué sur Caltech (officiel) via yolo val split=test

testé en cross-dataset Caltech → INRIA (meilleure configuration)

utilisé pour DeepSORT et ByteTrack sur KITTI.

::contentReference[oaicite:0]{index=0}
---

# Contact

Pour toute question concernant le projet, vous pouvez contacter :

**Ousmane Maiga**  
**eom6713@umoncton.ca**

