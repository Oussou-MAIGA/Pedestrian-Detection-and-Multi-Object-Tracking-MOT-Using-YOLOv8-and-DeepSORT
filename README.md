# README – Détection et Suivi de Piétons (YOLOv8s + DeepSORT)

## 1. Informations d’exécution (Cluster Trilium)

- JobName : yolo_caltech  
- Nœud utilisé : trig0042  
- GPUs utilisés : 4  
- CPU alloué : 1 (96 disponibles par nœud dans compute_full_node)  
- Temps d’exécution total : 2 h 05 min 41 s  
- Limite fixée : 7 h  
- ExitCode : 0:0  
- Script exécuté : train_yolo.slurm  
- Journaux :  
  - StdOut : /home/ousman/links/scratch/pieton_tm/modeles/yolov8/yolo_caltech_<jobID>.out  
  - StdErr : /home/ousman/links/scratch/pieton_tm/modeles/yolov8/yolo_caltech_<jobID>.err  

---

## 2. Présentation du projet

Ce projet traite de la détection et du suivi de piétons à l’aide de YOLOv8s pour la détection et DeepSORT pour le suivi multi-objet.

Le modèle final utilisé est :

best.pt  
Un modèle YOLOv8s pré-entraîné COCO puis fine-tuné sur Caltech.

Ce modèle est évalué :  
- sur le test Caltech (sets 06–10),  
- en cross-dataset sur INRIA.

---

## 3. Organisation du projet

```
projet_detection_suivi_pietons/
├── README.md
├── train_yolo.slurm
├── README_logs
├── track_ReID_deepsort.py
├── config/
│   ├── data_caltech.yaml
│   ├── data_inria.yaml
│   └── liste_chemin_image.sh
├── models/
│   └── best.pt
└── logs/
    ├── yolo_caltech_115655.out
    └── yolo_caltech_115655.err

```

## 4. Entraînement YOLOv8s sur Caltech

L’entraînement s’effectue via train_yolo.slurm.

### Modèle utilisé
- yolov8s.pt (pré-entraîné COCO)

### Dataset
- Caltech Pedestrian  
- Splits générés :  
  - train : sets 00–04  
  - val : set 05  
  - test : sets 06–10  

### Commande

```
yolo detect train   model="$MODEL"   data="$CFG/data.yaml"   imgsz=640 epochs=40 batch=16 device=0,1,2,3 workers=8   cache=False amp=False verbose=True   project="$PROJECT_OUT"   name=caltech_person
```

Le modèle final best.pt est généré automatiquement.

---

## 5. Résultats

### A. Évaluation Caltech → Caltech

Commande :

```
yolo detect val   model=models/best.pt   data=config/data_caltech.yaml   split=test   imgsz=640
```

Résultat :

- mAP@50 = 0.483

### B. Cross-dataset Caltech → INRIA

Commande :

```
yolo detect val   model=models/best.pt   data=config/data_inria.yaml   split=test   imgsz=640   name=caltech_to_inria_test
```

Résultat :

- mAP@50 = 0.689

---

## 6. Suivi de piétons (DeepSORT)

track_ReID_deepsort.py effectue le suivi multi-objet à partir des détections YOLO.

### Arguments

- --img_dir  
- --dets_dir  
- --out_dir  
- --embedder (défaut : mobilenet)  
- --max_age (défaut : 10)  
- --n_init (défaut : 3)  
- --max_cosine_distance (défaut : 0.4)

### Exemple 1 – paramètres par défaut

```
python track_ReID_deepsort.py ^
  --img_dir  "C:\Users\technicien\...\kitti_tracking\training\image_02\0012" ^
  --dets_dir "C:\Users\technicien\...\runs\detect\predict\0012\labels" ^
  --out_dir  "C:\Users\technicien\...\Caltech_Pedestrian\ousmane\0012"
```

### Exemple 2 – paramètres personnalisés

```
python track_ReID_deepsort.py   --img_dir  /chemin/kitti/image_02/0012   --dets_dir /chemin/predictions/0012/labels   --out_dir  /chemin/tracking/0012   --embedder mobilenet   --max_age 10   --n_init 3   --max_cosine_distance 0.4
```

Le script génère des images annotées et des fichiers YOLO contenant classe cx cy w h track_id.

---

## 7. Conclusion

- Le modèle YOLOv8s fine-tuné sur Caltech constitue le meilleur modèle.  
- Caltech → Caltech : mAP@50 = 0.483  
- Caltech → INRIA : mAP@50 = 0.689  
- Le domain gap explique pourquoi INRIA → Caltech est faible.  
- DeepSORT complète la détection par un suivi multi-image robuste.

---

## 8. Reproductibilité

### Générer les splits Caltech
```
bash config/liste_chemin_image.sh
```

### Entraîner sur Caltech
```
sbatch train_yolo.slurm
```

### Cross-dataset
```
yolo detect val   model=models/best.pt   data=config/data_inria.yaml   split=test
```

### Suivi DeepSORT
```
python track_ReID_deepsort.py --img_dir ... --dets_dir ... --out_dir ...
```
