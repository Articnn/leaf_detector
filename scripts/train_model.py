import os
import tensorflow as tf
from tensorflow.keras import layers, models
#from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt



# === Parámetros ===
DATA_DIR = "./data/leaves"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Tamaño más pequeño para mejor sensibilidad en datasets chicos, a lo mejor lo puedo aumentar un poco mas 
EPOCHS = 10
MODEL_PATH = "./models/leaf_model.keras"

# === 1. Cargar dataset ===
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(" Clases detectadas:", class_names)
for i, name in enumerate(class_names):
    print(f" Índice {i}: {name}")

# === 2. Preprocesamiento y Augmentación ===
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# Normalización integrada al modelo
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === 3. Modelo base ===
base_model = EfficientNetB0(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

# Fine-tuning desde capa 50 ahorita la movi a 60  entrenar por etapas primero apago y despues lo entreno con tunning que sea seguido osea dos entrenamientos 
#fine_tune_at = 80
#for layer in base_model.layers[:fine_tune_at]:
    #layer.trainable = False

# === 4. Modelo final ===
model = models.Sequential([
    normalization_layer,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])



model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 5. Callbacks ===
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)
]

# === 6. Entrenamiento ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === 7. Evaluar modelo ===
loss, accuracy = model.evaluate(val_ds)
print(f" Precisión en validación: {accuracy:.2%}")

# === 8. Gráficas ===
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Precisión por época")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Entrenamiento')
plt.plot(epochs_range, val_loss, label='Validación')
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Pérdida por época")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()


#----9 matriz de confusion para checar donde estan los fallos --- 



# === 9. Matriz de confusión ===



# Obtener predicciones reales vs predichas

