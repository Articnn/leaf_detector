# train_stage1.py
#se congelan los pesos del modelo base no los ajusto 
#adaptar el modelo que esta preentrenado con imagenet 
#clasificar hojas
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory

# === Configuración ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
DATA_DIR = "./data/leaves"
MODEL_PATH = "./models/leaf_stage1.keras"

# === Dataset ===
train_ds = image_dataset_from_directory(DATA_DIR, validation_split=0.2, subset="training",
                                        seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = image_dataset_from_directory(DATA_DIR, validation_split=0.2, subset="validation",
                                      seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
class_names = train_ds.class_names

# === Normalización + Aumento de datos ===
#data augmentation para hacer que el modelo vea variaciones realistas 
#es para ayudar a generalizar y evitar el sobre entrenamiiento
#
AUTOTUNE = tf.data.AUTOTUNE
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])
train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), tf.one_hot(y, len(class_names)))) # convertir las etiquetas en one-hot encoding como usa categoricalcrosssentory no sparse
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === Modelo base congelado ===
#son cargas efficientNetB0, es lo de pesos preentreandos
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)) # solo agregue las de arriba 
base_model.trainable = False

# === Modelo final ===
model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), #apaga de manera aleatoria el 50 % de neuronas para evitar el overfiting
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # no sobre confiar, 
    metrics=['accuracy']
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save(MODEL_PATH)
print(f"✅ Modelo base entrenado guardado en {MODEL_PATH}")
