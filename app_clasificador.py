import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
from keras.models import load_model

# Título de la app
st.title("Clasificador de Perros y Gatos")

# Cargar modelo entrenado
@st.cache_resource
def load_trained_model():
    model_path = "models/cats_and_dogs_model_5.keras"
    
    if not os.path.exists(model_path):
        file_id = "1-mSWToGWgEu706iTwXTha4hFhOq6aLmK"  # <- pon aquí tu ID real
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    return load_model(model_path)

model = load_trained_model()

# Tamaño esperado por el modelo
IMG_SIZE = (224, 224)

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen de un perro o un gato", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer y mostrar imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar imagen sin OpenCV
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = model.predict(img)[0][0]

    # Mostrar resultado
    if pred > 0.5:
        st.markdown(f"### 🐶 Es un **perro** con una probabilidad de {pred:.2f}")
    else:
        st.markdown(f"### 🐱 Es un **gato** con una probabilidad de {1 - pred:.2f}")