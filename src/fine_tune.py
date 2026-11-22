import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path.home() / "OneDrive" / "Documents" / "PERISHGUARD_MODEL_DATASET"

print("\n========== PERISHGUARDAI FINE-TUNING ==========")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "val",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    shuffle=False
)

if train_ds.class_names != val_ds.class_names:
    raise ValueError("Class order mismatch!")

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model(
    ROOT / "models" / "perishguard_cnn_backup.keras"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

output_path = ROOT / "models" / "perishguard_cnn_finetuned.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        output_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)

print("\n========== FINE-TUNING COMPLETED ==========")
print("Best model saved at:", output_path)
