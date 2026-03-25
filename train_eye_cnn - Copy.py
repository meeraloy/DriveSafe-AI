"""
DriveSafe AI — CNN Eye Classifier Training Script
===================================================
Dataset:  MRL Eye Dataset  (https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
          OR CEW eye patches (https://parnec.nuaa.edu.cn/xtan/ClosedEyeDatabases.html)

Expected folder structure after downloading:
    dataset/
        open/       ← images of open eyes
        closed/     ← images of closed eyes

Run:
    pip install tensorflow opencv-python scikit-learn matplotlib
    python train_eye_cnn.py

Output:
    eye_cnn_model.h5   ← saved model (load this in drivesafe_upgraded.py)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"          # folder with open/ and closed/ subfolders
IMG_SIZE     = (64, 64)           # resize all eyes to this
BATCH_SIZE   = 32
EPOCHS       = 30
MODEL_OUTPUT = "eye_cnn_model.h5"
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(dataset_dir):
    """Load images from open/ and closed/ subfolders, return X, y arrays."""
    X, y = [], []
    class_map = {"open": 1, "closed": 0}   # 1 = open, 0 = closed

    for label_name, label_int in class_map.items():
        folder = os.path.join(dataset_dir, label_name)
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Folder '{folder}' not found.\n"
                f"Make sure your dataset has subfolders: open/ and closed/"
            )
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.pgm'))]
        print(f"  Loading {len(files)} images from '{label_name}/'...")
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_int)

    X = np.array(X, dtype=np.float32) / 255.0   # normalize to [0, 1]
    X = X[..., np.newaxis]                        # add channel dim → (N, 64, 64, 1)
    y = np.array(y, dtype=np.int32)
    return X, y


def build_cnn(input_shape=(64, 64, 1)):
    """
    Lightweight CNN — fast enough for a laptop webcam.
    Architecture:
        Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → MaxPool
        → Flatten → Dense(128) → Dropout → Dense(1, sigmoid)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')   # 1 = open, 0 = closed
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training(history):
    """Plot accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='Train accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='Train loss')
    ax2.plot(history.history['val_loss'], label='Val loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=120)
    plt.show()
    print("Training curves saved → training_curves.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== DriveSafe AI — CNN Eye Classifier Training ===\n")

    # 1. Load data
    print("Loading dataset...")
    X, y = load_dataset(DATASET_DIR)
    print(f"  Total samples: {len(X)}  |  Open: {y.sum()}  |  Closed: {(y==0).sum()}\n")

    # 2. Train/val/test split  (70 / 15 / 15)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test   = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

    # 3. Data augmentation (only on training set)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # 4. Build model
    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    model.summary()

    # 5. Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(
            MODEL_OUTPUT, monitor='val_accuracy',
            save_best_only=True, verbose=1)
    ]

    # 6. Train
    print("\nTraining...\n")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1
    )

    # 7. Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Closed", "Open"]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # 8. Save curves
    plot_training(history)

    print(f"\nModel saved → {MODEL_OUTPUT}")
    print("Next step: use this model in drivesafe_upgraded.py")
