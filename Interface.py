import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Classification de Tumeurs Cérébrales")

@st.cache_resource
def load_model_cached():
    return load_model('best_model(1).h5')

model = load_model_cached()

uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image_rgb, width=300)
    
    if st.button("Prédire"):
        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
       
        predictions = model.predict(img_batch, verbose=0)
        class_names = ['Glioma', 'Meningioma', 'Pas de Tumeur', 'Pituitary']
        
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
        
        st.success(f"Prédiction: {predicted_class}")
        st.info(f"Confiance: {confidence:.2f}%")
        
        st.write("Probabilités:")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {predictions[0][i]*100:.2f}%")