import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# === Configuración ===
MODEL_PATH = './models/leaf_finetuned.keras'
IMG_PATH = './data/leaves/class_7/l7nr072.png'  # ⚠️ Cambia esta ruta por la hoja que quieras probar
IMG_SIZE = (224, 224)
DATA_DIR = './data/leaves'

# === Obtener clases en el mismo orden que entrenamiento ===
class_names = sorted(os.listdir(DATA_DIR))
print("📁 Clases detectadas:", class_names)

# === Cargar modelo ===
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

# === Cargar y preprocesar la imagen ===
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) # Normalizar igual que en entrenamiento NO NORMLAIZAR
img_array = np.expand_dims(img_array, axis=0)  # Convertir a forma (1, 224, 224, 3)

# === Hacer predicción ===
predictions = model.predict(img_array)  # te da el vector de probabilidades, 
pred_index = np.argmax(predictions[0])  # te da el indice con mayor probabilidad 
predicted_class = class_names[pred_index]
confidence = predictions[0][pred_index]

# === Mostrar resultado ===
plt.imshow(img)
plt.title(f'Predicción: {predicted_class} ({confidence*100:.2f}%)')
plt.axis('off')
plt.show()

print(f"🔍 Clase predicha: {predicted_class}")
print(f"📊 Confianza: {confidence:.4f}")
