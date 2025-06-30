import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Título de la app
st.title("Clasificador de Perros y Gatos")

# Cargar modelo entrenado

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="models/cats_and_dogs_model.tflite")
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_tflite_model()

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

    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        expected_shape = tuple(input_details[0]['shape'])
        expected_dtype = input_details[0]['dtype']

        img = img.astype(expected_dtype)

        if img.shape != expected_shape:
            st.error(f"❌ La imagen procesada tiene forma {img.shape}, pero se esperaba {expected_shape}")
        else:
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Mostrar resultado
            if pred > 0.5:
                st.markdown(f"### 🐶 Es un **perro** con una probabilidad de {pred:.2f}")
            else:
                st.markdown(f"### 🐱 Es un **gato** con una probabilidad de {1 - pred:.2f}")
    except Exception as e:
        st.error(f"❌ Error durante la inferencia: {e}")