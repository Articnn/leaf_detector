import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURACIÓN ===
MODEL_PATH = "./models/leaf_finetuned.keras"
IMG_SIZE = (224, 224)
DATA_DIR = "./data/leaves"
OUTPUT_CSV = "errores_prediccion.csv"

# === 1. Cargar modelo ===
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

# === 2. Cargar clases ===
class_names = sorted(os.listdir(DATA_DIR))
print("📁 Clases detectadas:", class_names)

# === 3. Recolectar datos ===
y_true, y_pred, confidences, filenames = [], [], [], []

for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_path):
        img_path = os.path.join(class_path, fname)

        # Cargar y preprocesar imagen
        img = image.load_img(img_path, target_size=IMG_SIZE)
        #img_array = image.img_to_array(img) / 255.0
        img_array = image.img_to_array(img)  # sin normalizar

        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        pred_index = np.argmax(preds[0])
        confidence = preds[0][pred_index]

        # Guardar resultados
        y_true.append(class_index)
        y_pred.append(pred_index)
        confidences.append(confidence)
        filenames.append((fname, class_name, class_names[pred_index]))

# === 4. Matriz de confusión ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format="d", xticks_rotation=45)
plt.title("🔍 Matriz de Confusión")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# === 5. CSV de errores ===
errores = []
for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        errores.append({
            "archivo": filenames[i][0],
            "clase_real": filenames[i][1],
            "clase_predicha": filenames[i][2],
            "confianza": f"{confidences[i]:.4f}"
        })

df = pd.DataFrame(errores)
df.to_csv(OUTPUT_CSV, index=False)
print(f"📝 CSV de errores guardado como {OUTPUT_CSV}")
