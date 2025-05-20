# train_stage2_finetune.py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# === Configuración ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15
FINE_TUNE_AT = 80
DATA_DIR = "./data/leaves"
MODEL_STAGE1 = "./models/leaf_stage1.keras"
MODEL_PATH = "./models/leaf_finetuned.keras"

# === Dataset ===
train_ds = image_dataset_from_directory(DATA_DIR, validation_split=0.2, subset="training",
                                        seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = image_dataset_from_directory(DATA_DIR, validation_split=0.2, subset="validation",
                                      seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
class_names = train_ds.class_names
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))

# === Cargar modelo base entrenado ===
model = load_model(MODEL_STAGE1)

# === Activar fine-tuning ===
base_model = model.layers[1]  # La EfficientNetB0 está en el índice 1
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False
for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
model.save(MODEL_PATH)
print(f"✅ Modelo fine-tuneado guardado en {MODEL_PATH}")
