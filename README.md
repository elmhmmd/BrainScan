# Classification de Tumeurs Cérébrales

Système de classification automatique de tumeurs cérébrales à partir d'images IRM utilisant un réseau de neurones convolutif (CNN).

## Description du Projet

Ce projet classifie quatre types de tumeurs cérébrales :
- Glioma
- Meningioma
- Pituitary
- No Tumor

Performance du modèle : 91% d'accuracy

## Structure des Fichiers

```
BrainScan/
│
├── Data/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
├── notebook.ipynb
├── Interface.py
├── report.py
├── best_model(1).h5
└── history.pkl
```

## Instructions pour l'Exécuter

### Installation

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn streamlit pillow
```

### Entraîner le Modèle

Ouvrir et exécuter Brainscan.ipynb

### Interface de Prédiction

```bash
streamlit run Interface.py
```

### Rapport Détaillé

```bash
streamlit run report.py
```
