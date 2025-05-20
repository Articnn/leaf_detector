import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from streamlit_lottie import st_lottie
from logic import procesar_mensaje
import time

# === CONFIGURACIÓN GENERAL ===
st.set_page_config(page_title="Leaf Prediction", page_icon="🌿")


# === CARGAR ANIMACIÓN ===
def cargar_animacion_local(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error("⚠️ No se pudo cargar la animación.")
        return None

if "ya_animado" not in st.session_state:
    animacion = cargar_animacion_local("./assetss/load.json")
    if animacion:
        st_lottie(animacion, height=300, key="loading")
        st.markdown("<h2 style='text-align: center;'>🌿 Cargando Clasificador de Hoja...</h2>", unsafe_allow_html=True)
        time.sleep(2.5)  # espera para que la animación se vea
    st.session_state.ya_animado = True


# === ESTILOS PERSONALIZADOS ===
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    section[data-testid="stSidebar"] { background-color: #171c26; }
    input[type="text"] { background-color: #1e222a; color: white; border: 1px solid #3a3f4b; }
    button[kind="secondary"] { background-color: #1c352d !important; color: white !important; border: none; }
    .chat-user { background-color: #1f2933; color: #b2f2bb; padding: 8px 12px; border-radius: 12px; margin: 4px 0; align-self: flex-end; max-width: 80%; }
    .chat-bot { background-color: #234d32; color: white; padding: 8px 12px; border-radius: 12px; margin: 4px 0; align-self: flex-start; max-width: 80%; }
    .chat-container { display: flex; flex-direction: column; }
    .leaf-image { box-shadow: 0 0 12px #2f8f46; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# === VARIABLES DE CONFIGURACIÓN ===
MODEL_PATH = "./models/leaf_finetuned.keras"
DATA_DIR = "./data/leaves"
SAMPLE_DIR = "./data/demo"
IMG_SIZE = (224, 224)
class_names = sorted(os.listdir(DATA_DIR))

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === SESIÓN INICIAL ===
for k, v in {
    "selected_image": None,
    "uploaded_image": None,
    "predicted_class": None,
    "nombre": None,
    "clase_actual": None,
    "chat_history": [],
    "input_guardado": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

tips = {
    "class_1": "Retira hojas afectadas y evita el riego por aspersión. Aplica fungicidas si es necesario.",
    "class_2": "Aumenta ventilación entre plantas. Usa tratamientos naturales como extracto de ajo.",
    "class_3": "Elimina hojas dañadas y mantén el suelo bien drenado.",
    "class_4": "Evita el exceso de nitrógeno. Desinfecta herramientas tras podar.",
    "class_5": "Aplica controles biológicos y evita mojar el follaje al regar.",
    "class_6": "Mantén una buena ventilación y poda zonas densas para evitar propagación.",
    "class_7": "Riega directo al suelo. Aplica fungicida si las manchas aumentan."
}

# === SIDEBAR ===
with st.sidebar:
    st.success("Modelo cargado correctamente")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        st.session_state.selected_image = None

    with st.expander("Seleccionar imagen de galería"):
        for idx, img_name in enumerate(sorted(os.listdir(SAMPLE_DIR))[:6]):
            img_path = os.path.join(SAMPLE_DIR, img_name)
            if st.button(f"Usar {img_name}", key=f"galeria_{img_name}"):
                st.session_state.selected_image = img_path
                st.session_state.uploaded_image = None
            st.image(img_path, width=80)

    st.markdown("---")
    st.markdown("### 💬 Chat de ayuda sobre tus plantas")

    if st.button("🧹 Limpiar chat"):
        st.session_state.chat_history = []
        st.session_state.nombre = None
        st.rerun()

    user_input = st.text_input("Escribe tu mensaje:", key="chat_input")

    if user_input and user_input != st.session_state.input_guardado:
        respuesta = procesar_mensaje(user_input, st.session_state)
        st.session_state.chat_history.append(("Tú", user_input))
        st.session_state.chat_history.append(("GreenBot 🌿", respuesta))
        st.session_state.input_guardado = user_input
        st.rerun()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for autor, mensaje in st.session_state.chat_history:
        clase = "chat-user" if autor == "Tú" else "chat-bot"
        st.markdown(f'<div class="{clase}"><b>{autor}:</b><br>{mensaje}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗨️ Recomendación para cuidado del árbol")
    st.info(tips.get(st.session_state.predicted_class, "Aún no se ha clasificado una hoja."))

# === PROCESO DE IMAGEN Y PREDICCIÓN ===
img = None
if st.session_state.uploaded_image:
    img = Image.open(st.session_state.uploaded_image).convert("RGB")
elif st.session_state.selected_image:
    img = Image.open(st.session_state.selected_image).convert("RGB")

if img:
    st.image(img, caption="Imagen cargada", use_container_width=True, output_format="auto", channels="RGB")

    img_array = image.img_to_array(img.resize(IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    confidence = preds[0][pred_index]
    predicted_class = class_names[pred_index]

    st.session_state.predicted_class = predicted_class

    st.markdown(f"### Predicción: **{predicted_class}**")
    st.markdown(f"Confianza: **{confidence * 100:.2f}%**")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Predicción: {predicted_class} ({confidence * 100:.2f}%)")
    ax.axis('off')
    st.pyplot(fig)
