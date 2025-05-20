import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# === CONFIGURACIÓN ===
MODEL_PATH = "./models/leaf_finetuned.keras"
IMG_SIZE = (224, 224)
class_names = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7']

# === Cargar modelo ===
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

# === Solicitar imagen al usuario ===
img_path = input("🖼️ Ingresa la ruta de la imagen (./models/prueba.jpeg): ").strip()

if not os.path.exists(img_path):
    print("❌ La ruta no existe. Verifica el nombre del archivo.")
    exit()

# === Cargar imagen ===
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array   # Normalizar si tu modelo lo requiere
img_array = np.expand_dims(img_array, axis=0)

# === Predicción ===
preds = model.predict(img_array, verbose=0)
pred_index = np.argmax(preds[0])
pred_class = class_names[pred_index]
confidence = preds[0][pred_index]

# === Mostrar imagen y resultado ===
plt.imshow(img)
plt.title(f"Predicción: {pred_class} ({confidence*100:.2f}%)")
plt.axis("off")
plt.show()

# También en consola
print(f"🔍 Clase predicha: {pred_class}")
print(f"📊 Confianza: {confidence:.4f}")
