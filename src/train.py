import json
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parent.parent
DATASET = Path.home() / "OneDrive" / "Documents" / "PERISHGUARD_MODEL_DATASET"

MODEL_DIR = ROOT / "models"
GRAPH_DIR = ROOT / "outputs" / "graphs"
REPORT_DIR = ROOT / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- SETTINGS ----------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

print("\n========== PERISHGUARDAI TRAINING ==========")
print("Dataset:", DATASET)

# ---------------- LOAD DATA ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names

if val_ds.class_names != class_names:
    raise ValueError("Training and validation class names do not match!")

num_classes = len(class_names)

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("\nClasses:", class_names)
print("Number of classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------- DATA AUGMENTATION ----------------
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# ---------------- CNN MODEL ----------------
inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
x = augmentation(x)

x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "perishguard_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.3,
        min_lr=0.000001,
        verbose=1
    )
]

# ---------------- TRAIN ----------------
print("\n========== STARTING TRAINING ==========\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- SAVE RESULTS ----------------
pd.DataFrame(history.history).to_csv(
    REPORT_DIR / "training_history.csv",
    index=False
)

# Training accuracy graph
plt.figure(figsize=(9, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("PerishGuardAI - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(GRAPH_DIR / "accuracy.png", dpi=150)
plt.close()

# Training loss graph
plt.figure(figsize=(9, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("PerishGuardAI - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(GRAPH_DIR / "loss.png", dpi=150)
plt.close()

print("\n========== TRAINING COMPLETED ==========")
print("Best model:", MODEL_DIR / "perishguard_cnn.keras")
print("Class names:", MODEL_DIR / "class_names.json")
print("Graphs:", GRAPH_DIR)
print("Training history:", REPORT_DIR / "training_history.csv")
