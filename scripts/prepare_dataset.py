import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# === Paso 1: Reorganizar carpetas ===
raw_folders = [f"./data/leaf{i}" for i in range(1, 8)]
target_root = "./data/leaves"

os.makedirs(target_root, exist_ok=True)

for idx, folder_path in enumerate(raw_folders):
    class_name = f"class_{idx+1}"
    class_path = os.path.join(target_root, class_name)
    os.makedirs(class_path, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(folder_path, file)
            dst = os.path.join(class_path, file)
            shutil.copy(src, dst)

print("✅ Carpetas reorganizadas correctamente.")

# === Paso 2: Cargar dataset ===
BATCH_SIZE = 16
IMG_SIZE = (224, 224)

dataset = image_dataset_from_directory(
    target_root,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print("📂 Clases detectadas:", class_names)

# === Paso 3: Visualizar 9 imágenes ===
plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.tight_layout()
plt.savefig("data/sample_grid.png")
plt.show()

# === Paso 4: Dividir en train/val ===
train_size = 0.8
total_batches = tf.data.experimental.cardinality(dataset).numpy()
train_batches = int(total_batches * train_size)

train_ds = dataset.take(train_batches)
val_ds = dataset.skip(train_batches)

print(f"✅ Dataset dividido: {train_batches} batches para entrenamiento")
