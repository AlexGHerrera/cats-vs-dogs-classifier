import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Parámetros constantes
IMG_SIZE = (224, 224)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fpvhMNW3tjX-s7eN5Wsg6DMl08iua-dv"

import requests

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    st.warning(f"Descargando modelo desde Google Drive... Esto puede tardar unos segundos.")
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Modelo descargado correctamente.")
        return True
    except Exception as e:
        st.error(f"❌ Error descargando el modelo: {e}")
        return False

# Cargar modelo TFLite con caché y manejo de errores
@st.cache_resource
def load_tflite_model():
    if not download_model():
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        st.success("✅ Modelo TFLite cargado correctamente")
        return interpreter
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return None

# Función para preprocesar imagen
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Función para hacer la predicción
def predict(interpreter, img: np.ndarray):
    if interpreter is None:
        st.error("❌ Modelo no cargado, no se puede predecir")
        return None

    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        expected_shape = tuple(input_details[0]['shape'])
        expected_dtype = input_details[0]['dtype']

        img = img.astype(expected_dtype)

        if img.shape != expected_shape:
            st.error(f"❌ La imagen procesada tiene forma {img.shape}, pero se esperaba {expected_shape}")
            return None

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        return pred
    except Exception as e:
        st.error(f"❌ Error durante la inferencia: {e}")
        return None

# Título
st.title("Clasificador de Perros y Gatos")

# Cargar modelo
interpreter = load_tflite_model()

if interpreter is not None:
    uploaded_file = st.file_uploader("Sube una imagen de un perro o un gato", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen cargada", use_container_width=True)
            img = preprocess_image(image)
            with st.spinner("Realizando predicción..."):
                pred = predict(interpreter, img)
            if pred is not None:
                if pred > 0.5:
                    st.markdown(f"### 🐶 Es un **perro** con una probabilidad de {pred:.2f}")
                else:
                    st.markdown(f"### 🐱 Es un **gato** con una probabilidad de {1-pred:.2f}")
        except Exception as e:
            st.error(f"❌ Error procesando la imagen: {e}")
else:
    st.warning("Carga fallida del modelo. No puedes hacer predicciones.")