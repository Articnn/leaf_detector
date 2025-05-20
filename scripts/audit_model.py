import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# === Configuración ===
MODEL_PATH = './models/leaf_model.h5'
DATA_DIR = './data/leaves'
IMG_SIZE = (224, 224)
class_names = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7']

# === Cargar modelo ===
model = tf.keras.models.load_model(MODEL_PATH)
print(" Modelo cargado correctamente")

# === Seleccionar una imagen aleatoria de cada clase ===
for class_name in class_names:
    class_path = os.path.join(DATA_DIR, class_name)
    img_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, img_name)

    # Cargar imagen
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predicción
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    # Mostrar resultado
    plt.imshow(img)
    plt.title(f'Real: {class_name}\nPredicción: {predicted_class} ({confidence*100:.2f}%)')
    plt.axis('off')
    plt.show()

    print(f" Clase real: {class_name}")
    print(f" Predicción: {predicted_class}")
    print(f" Confianza: {confidence:.4f}")
    print("=" * 40)
