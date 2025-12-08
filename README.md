# Détection et Suivi de Piétons  
**Haar/SVM · HOG/SVM · YOLOv8s · DeepSORT · ByteTrack**

Projet du cours — Université de Moncton  
Auteur : **Ousmane Maiga**  
Superviseur : **Pr. Moulay Akhloufi – PRIME Lab**

---

# 1. Description du projet

Ce projet compare trois approches de **détection de piétons** :

- Haar + SVM  
- HOG + SVM  
- YOLOv8s (meilleur modèle)

et deux algorithmes de **suivi multi-objets** :

- DeepSORT (avec ReID MobileNet)  
- ByteTrack  

Objectifs :

- analyser pourquoi les détecteurs classiques échouent en scène réelle  
- étudier la généralisation cross-dataset (Caltech ↔ INRIA)  
- mesurer l’impact de la qualité des détections sur le tracking  
- produire des résultats visuels + deux vidéos finales de suivi  

---

## 2. Structure du projet

```text
projet_detection_suivi_pietons/
├── README.md
├── train_yolo.slurm
├── images/
│   ├── haar_caltech_1.png
│   ├── haar_caltech_2.png
│   ├── haar_inria_1.png
│   ├── haar_inria_2.png
│   ├── hog_inria_1.png
│   ├── hog_inria_2.png
│   ├── yolo_inria_1.jpg
│   └── yolo_inria_2.jpg
├── videos/
│   ├── DeepSort.mp4
│   └── ByteTrack.mp4
├── models/
│   └── caltech.pt       ← meilleur modèle YOLOv8s (entraîné sur Caltech)
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
│   ├── convert_pred_to_MOT.py
│   ├── convert_kitti_GT_to_MOT.py
│   ├── extract_images.py
│   ├── extract_annotations.py
│   ├── convertir_vbb.py
│   ├── video_to_frames.py
│   └── images_to_videos.py
│── config/
│   ├── data_caltech.yaml
│   ├── data_inria.yaml
│   └── liste_chemin_image.sh
└── README_logs
```
---

# 3. Téléchargement des datasets (liens officiels)

Télécharger les datasets depuis les liens officiels car trop volumineux :

### 🔹 **Caltech Pedestrian Dataset**  
https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

### 🔹 **INRIA Person Dataset**  
https://github.com/olt/inria-object-detection

### 🔹 **KITTI Tracking Dataset**  
https://www.cvlibs.net/datasets/kitti/eval_tracking.php

Créer ensuite :

datasets/Caltech/
datasets/INRIA/
datasets/KITTI/


---

# 4. Exemples de détection 

### Haar + SVM (Caltech - Caltech & INRIA - INRIA)

<p align="center">
  <img src="images/haar_caltech_1.png" width="260" />
  <img src="images/haar_caltech_2.png" width="260" />
</p>

<p align="center">
  <img src="images/haar_inria_1.png" width="260" />
  <img src="images/haar_inria_2.png" width="260" />
</p>

### HOG + SVM (INRIA - INRIA)

<p align="center">
  <img src="images/hog_inria_1.png" width="260" />
  <img src="images/hog_inria_2.png" width="260" />
</p>

### YOLOv8s (Caltech - INRIA)

<p align="center">
  <img src="images/yolo_caltech_inria_1.jpg" width="260" />
  <img src="images/yolo_caltech_inria_2.jpg" width="260" />
</p>

---

# 5. Résultats de suivi 

Les vidéos sont dans :  
videos/DeepSort.mp4 et
videos/ByteTrack.mp4


### DeepSORT

[Voir la vidéo DeepSORT](videos/DeepSort.mp4)

### ByteTrack

[Voir la vidéo ByteTrack](videos/ByteTrack.mp4)

---

### Modèle de base utilisé pour l’entraînement

Le fine-tuning YOLOv8s sur Caltech part du modèle pré-entraîné COCO :

**Modèle de base : `yolov8s.pt`**  
Téléchargement officiel :  
https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt

Ce modèle est chargé automatiquement dans `train_yolo.slurm` via :

MODEL=“yolov8s.pt”

Il sert de point de départ pour obtenir notre modèle final **fine-tuné sur Caltech**, nommé :
modeles/caltech_person/weights/best.pt

Ce modèle fine-tuné est ensuite utilisé dans :
- l’évaluation Caltech → Caltech  
- l’évaluation Caltech → INRIA  
- la génération des prédictions pour le tracking DeepSORT et ByteTrack

# 6. Entraînement YOLOv8s (Caltech)

### SLURM (Cluster Trilium)
sbatch projet_detection_suivi_pietons/train_yolo.slurm


### Informations d’exécution  
- GPU : 4  
- Temps total : **2 h 05 min 41 s**  
- Modèle obtenu → **modeles/yolov8s/caltech_person/weights/best.pt**

---

# 7. Évaluation YOLOv8s

### ✔ Caltech → INRIA


yolo detect val model=modeles/caltech_person/weights/best.pt data=config/data_inria.yaml split=test

→ **mAP@50 = 0.689**

---

# 8. Suivi DeepSORT

python track_ReID_deepsort.py
--img_dir path/imgs
--dets_dir path/yolo_detection
--out_dir output/
--embedder mobilenet
--max_age 10 --n_init 3 --max_cosine_distance 0.4


Sorties :
- images annotées  
- labels YOLO + track_id

---

# 9. Reproductibilité complète

## (1) Télécharger datasets  
→ Voir section 3

## (2) Convertir Caltech (.seq + .vbb → images + YOLO)
python convertir_vbb.py
python extract_images.py
python extract_annotations.py

## (3) Générer splits
bash config/liste_chemin_image.sh

## (4) Entraîner YOLO
sbatch train_yolo.slurm
dos2unix train_yolo.slurm


## (5) Suivi multi-objet avec DeepSORT (MobileNet ReID)

Le suivi DeepSORT n’est pas appelé directement via Ultralytics.  
On utilise le script `track_ReID_deepsort.py`, qui prend en entrée :

- les **images** (frames KITTI),
- les **détections YOLOv8s** (fichiers `.txt` au format YOLO, générés avec `save_txt=True`),
- un **dossier de sortie** pour les images annotées + les labels avec ID de piste.

Le ReID utilisé est une variante légère basée sur **MobileNet**, ce qui permet d’avoir un suivi plus stable qu’avec ByteTrack tout en restant rapide.

#### a) Pré-requis

1. Avoir généré les détections YOLOv8s sur KITTI, par exemple :

```bash
yolo detect predict \
  model=modeles/caltech_person/weights/best.pt \
  source=datasets/KITTI/images/training/image_02/0019 \
  imgsz=1408 \
  conf=0.60 \
  save=True \
  save_txt=True \
  project=runs/detect \
  name=kitti_0019_yolo

Cela crée un dossier de ce type :

runs/detect/kitti_0019_yolo/
├── 000000.png
├── 000001.png
├── ...
└── labels/
    ├── 000000.txt
    ├── 000001.txt
    └── ...

	•	datasets/KITTI/images/training/image_02/0019/ : frames originales KITTI
	•	runs/detect/kitti_0019_yolo/labels/ : fichiers YOLO (cls cx cy w h conf)

b) Commande DeepSORT (exemple)

python track_ReID_deepsort.py \
  --img_dir  datasets/KITTI/images/training/image_02/0019 \
  --dets_dir runs/detect/kitti_0019_yolo/labels \
  --out_dir  runs/tracking/deepsort_0019 \
  --embedder mobilenet \
  --max_age 10 \
  --n_init 3 \
  --max_cosine_distance 0.4

•	--img_dir : dossier des images KITTI pour une séquence (par ex. 0019)
	•	--dets_dir : dossiers des fichiers .txt YOLO générés par Ultralytics
	•	--out_dir : dossier de sortie où seront enregistrés les résultats DeepSORT
	•	--embedder : type de ReID utilisé (ici mobilenet)
	•	--max_age : nombre de frames pendant lesquelles une piste peut survivre sans détection
	•	--n_init : nombre de frames consécutives nécessaires pour valider une nouvelle piste
	•	--max_cosine_distance : seuil pour la distance d’apparence (ReID)

c) Résultats générés
Le script produit :

runs/tracking/deepsort_0019/
├── frames/
│   ├── 000000.png         # image annotée avec ID de piste
│   ├── 000001.png
│   └── ...
└── labels/
    ├── 000000.txt         # cls cx cy w h track_id
    ├── 000001.txt
    └── ...
 ...
## (6) Suivi avec ByteTrack (Ultralytics)

Ultralytics intègre directement ByteTrack dans la commande yolo track.
Cette méthode permet d’exécuter le suivi multi-objet sans script additionnel, en utilisant le meilleur modèle YOLOv8s entraîné sur Caltech.

Commande d’exécution (exemple : séquence KITTI 0019)


yolo track \
  model="modeles/caltech_person/weights/best.pt" \
  source="datasets/KITTI/images/training/0019" \
  imgsz=1408 \
  conf=0.60 \
  save=True \
  save_txt=True \
  save_json=True \
  tracker="bytetrack.yaml" \
  project="runs/kitti_eval" \
  name="bytetrack_0019"

Description
	•	model= : utilise le modèle YOLOv8s fine-tuné sur Caltech.
	•	source= : dossier contenant les images de la séquence KITTI.
	•	tracker="bytetrack.yaml" : active le suivi ByteTrack.
	•	save=True : enregistre une vidéo annotée (format .mp4).
	•	save_txt=True : exporte les identités au format YOLO (cls cx cy w h track_id).
	•	save_json=True : génère un fichier compatible MOTChallenge.

Sorties générées

La commande produit automatiquement :
	•	une vidéo annotée avec les IDs des piétons ;
	•	un dossier labels/ contenant les résultats image par image ;
	•	un fichier JSON compatible MOT pour l’évaluation.

Ces fichiers sont regroupés dans :

runs/kitti_eval/bytetrack_0019/

Ce qui permet ensuite :
	•	de convertir en format MOT avec convert_Pred_to_MOT.py,
	•	puis d’évaluer avec eval_MOT.py (IDF1, MOTA, etc.).
---

# 10. Modèle final utilisé

modeles/caltech_person/weights/best.pt

---