import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Rapport du Projet", layout="wide")

st.markdown("# Rapport du Projet : Classification de Tumeurs Cérébrales")

# Load actual data
@st.cache_resource
def load_model_data():
    try:
        model = load_model('best_model(1).h5')
        return model
    except:
        return None

@st.cache_data
def load_history():
    try:
        with open('history.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model_data()
history = load_history()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Introduction", 
    "🔄 Prétraitement", 
    "🏗️ Architecture", 
    "📊 Résultats", 
    "💡 Conclusion"
])

with tab1:
    st.markdown("## Introduction")
    
    st.markdown("""
    ### Objectif du Projet
    Développer un système de classification automatique de tumeurs cérébrales à partir d'images IRM 
    utilisant un réseau de neurones convolutif (CNN).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Dataset
        - **Source**: Images IRM de tumeurs cérébrales
        - **Nombre total d'images**: 8,000 images (après augmentation)
        - **Format**: JPEG, PNG
        - **Résolution**: Redimensionnées à 224x224 pixels
        """)
    
    with col2:
        st.markdown("""
        ### Classes
        1. **Glioma** (2000 images)
        2. **Meningioma** (2000 images)
        3. **Pituitary** (2000 images)
        4. **No Tumor** (2000 images)
        """)
    
    st.markdown("---")
    
    st.markdown("### Distribution Initiale des Données")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    initial_counts = [1621, 1645, 1757, 2000]
    ax.bar(classes, initial_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_ylabel('Nombre d\'images')
    ax.set_title('Distribution Initiale des Images par Classe')
    ax.axhline(y=2000, color='r', linestyle='--', label='Objectif (2000)')
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.markdown("## Prétraitement des Données")
    
    st.markdown("""
    ### 1. Nettoyage du Dataset
    - Vérification des extensions de fichiers valides (`.jpeg`, `.jpg`, `.bmp`, `.png`)
    - Suppression des fichiers non conformes
    """)
    
    with st.expander("📝 Code de nettoyage"):
        st.code("""
valid_extensions = ['.jpeg', '.jpg', '.bmp', '.png']

for subfolder in os.listdir(images_folder):
    subfolder_path = os.path.join(images_folder, subfolder)
    if os.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            _, extension = os.path.splitext(file_path)
            if extension.lower() not in valid_extensions:
                os.remove(file_path)
        """, language='python')
    
    st.markdown("---")
    
    st.markdown("""
    ### 2. Mélange des Données
    - Combinaison de toutes les images et leurs labels
    - Mélange aléatoire pour éviter les biais d'ordre
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 3. Redimensionnement
    - Toutes les images redimensionnées à **224x224 pixels**
    - Uniformisation pour l'entrée du CNN
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 4. Rééquilibrage des Classes
    
    **Problème identifié**: Déséquilibre dans la distribution des classes
    
    **Solution appliquée**: Data Augmentation avec ImageDataGenerator
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Techniques d'Augmentation")
        st.markdown("""
        - Rotation aléatoire (±15°)
        - Décalage horizontal/vertical (10%)
        - Retournement horizontal
        - Zoom aléatoire (10%)
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        final_counts = [2000, 2000, 2000, 2000]
        ax.bar(classes, final_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Nombre d\'images')
        ax.set_title('Distribution Finale (Équilibrée)')
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.markdown("""
    ### 5. Normalisation
    - Conversion des valeurs de pixels de [0, 255] à [0, 1]
    - Améliore la convergence du modèle
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 6. Encodage des Labels
    - Transformation des labels textuels en format numérique
    - Utilisation de **One-Hot Encoding**
    """)

with tab3:
    st.markdown("## Architecture du Modèle CNN")
    
    st.markdown("### Structure du Réseau")
    
    architecture_data = {
        'Couche': [
            'Conv2D (1)', 'MaxPooling2D (1)',
            'Conv2D (2)', 'MaxPooling2D (2)',
            'Conv2D (3)', 'MaxPooling2D (3)',
            'Flatten', 'Dropout',
            'Dense (1)', 'Dense (2)'
        ],
        'Paramètres': [
            '32 filtres, 3x3, ReLU',
            '2x2',
            '32 filtres, 3x3, ReLU',
            '2x2',
            '64 filtres, 3x3, ReLU',
            '2x2',
            '-',
            '50%',
            '32 neurones, ReLU',
            '4 neurones, Softmax'
        ],
        'Output Shape': [
            '(222, 222, 32)',
            '(111, 111, 32)',
            '(109, 109, 32)',
            '(54, 54, 32)',
            '(52, 52, 64)',
            '(26, 26, 64)',
            '(43264)',
            '(43264)',
            '(32)',
            '(4)'
        ],
        'Params': [
            '896',
            '0',
            '9,248',
            '0',
            '18,496',
            '0',
            '0',
            '0',
            '1,384,480',
            '132'
        ]
    }
    
    df_arch = pd.DataFrame(architecture_data)
    st.dataframe(df_arch, use_container_width=True)
    
    st.markdown("---")
    
    if model:
        st.markdown("### Résumé du Modèle")
        
        # Get model summary as string
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        model.summary()
        summary_string = buffer.getvalue()
        sys.stdout = old_stdout
        
        st.code(summary_string, language=None)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Paramètres du Modèle
        - **Total de paramètres**: 1,413,252
        - **Paramètres entraînables**: 1,413,252
        - **Taille du modèle**: 5.39 MB
        """)
    
    with col2:
        st.markdown("""
        ### Configuration de l'Entraînement
        - **Optimizer**: Adam (lr=0.001)
        - **Loss Function**: Categorical Crossentropy
        - **Métrique**: Accuracy
        - **Batch Size**: 32
        - **Epochs**: 50 (avec Early Stopping)
        """)

with tab4:
    st.markdown("## Résultats et Performance")
    
    st.markdown("### Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy Test", "91%")
    
    with col2:
        st.metric("Durée d'Entraînement", "440 sec")
    
    with col3:
        st.metric("Epochs Exécutés", "17/50")
    
    with col4:
        st.metric("Meilleure Val Accuracy", "92.27%")
    
    st.markdown("---")
    
    if history:
        st.markdown("### Courbes d'Apprentissage")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(len(history['accuracy']))
        
        # Accuracy
        ax1.plot(epochs, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Evolution of Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Evolution of Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Fichier history.pkl non trouvé. Sauvegardez l'historique avec: pickle.dump(history.history, open('history.pkl', 'wb'))")
    
    st.markdown("---")
    
    st.markdown("### Rapport de Classification Détaillé")
    
    classification_data = {
        'Classe': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'Precision': [0.90, 0.85, 0.97, 0.92],
        'Recall': [0.87, 0.85, 0.96, 0.95],
        'F1-Score': [0.89, 0.85, 0.97, 0.93],
        'Support': [399, 413, 404, 384]
    }
    
    df_class = pd.DataFrame(classification_data)
    st.dataframe(df_class, use_container_width=True)
    
    st.markdown("---")
    

with tab5:
    st.markdown("## Conclusion et Perspectives")
    
    st.markdown("### Résultats Obtenus")
    
    st.success("""
    ✅ **Accuracy globale de 91%** sur l'ensemble de test
    
    ✅ **Modèle équilibré** grâce à l'augmentation de données
    
    ✅ **Convergence rapide** avec Early Stopping (17 epochs)
    
    ✅ **Performance stable** entre entraînement et validation
    """)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p><b>Projet de Classification de Tumeurs Cérébrales</b> | 2025</p>
</div>
""", unsafe_allow_html=True)