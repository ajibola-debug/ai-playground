"""Real vs AI-generated image detector on CIFAKE, in TensorFlow/Keras. One file.

Folder layout it expects (CIFAKE as it comes):
    cifake/train/REAL   cifake/train/FAKE
    cifake/test/REAL    cifake/test/FAKE

Install:
    pip install tensorflow numpy scikit-learn matplotlib

Run (trains, then evaluates on the test set):
    python cifake_tensorflow.py --data-dir ~/Desktop/cifake
    python cifake_tensorflow.py --data-dir ~/Desktop/cifake --backbone mobilenet --epochs 3
    python cifake_tensorflow.py --data-dir ~/Desktop/cifake --epochs 1 --subset 4000   # quick check

Models:
    cnn        small CNN trained from scratch at 32x32. Fast on a laptop CPU.
    mobilenet  MobileNetV2 pretrained on ImageNet, fine-tuned at 96x96. Slower; better with a GPU.

Outputs:
    models_tf/best_<backbone>.keras         trained model
    results_tf/metrics_<backbone>.json      accuracy, precision, recall, F1, ROC-AUC (+ JPEG tests)
    results_tf/confusion_matrix_<backbone>.png
    results_tf/history_<backbone>.json      loss/accuracy per epoch
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ["REAL", "FAKE"]  # REAL = 0, FAKE = 1 (FAKE is the positive class)
IMG = 32                        # every CIFAKE image is 32x32
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------- data

def load_folder(folder, batch_size, seed, **kwargs):
    return keras.utils.image_dataset_from_directory(
        folder, labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
        image_size=(IMG, IMG), batch_size=batch_size, shuffle=True, seed=seed, **kwargs)


def limit(ds, n_images, batch_size):
    return ds if n_images is None else ds.take(max(1, n_images // batch_size))


def train_val_data(data_dir, batch_size, subset, seed, val_fraction=0.1):
    folder = os.path.join(data_dir, "train")
    train = load_folder(folder, batch_size, seed, validation_split=val_fraction, subset="training")
    val = load_folder(folder, batch_size, seed, validation_split=val_fraction, subset="validation")
    val_n = None if subset is None else int(subset * val_fraction)
    return in_memory(limit(train, subset, batch_size)), in_memory(limit(val, val_n, batch_size))


def in_memory(ds):
    """Decode the image files once, keep them in RAM as uint8 (about 300 MB for all of CIFAKE),
    so every epoch after the first skips reading from disk."""
    to_uint8 = lambda x, y: (tf.cast(x, tf.uint8), y)
    to_float = lambda x, y: (tf.cast(x, tf.float32), y)
    return ds.map(to_uint8, num_parallel_calls=AUTOTUNE).cache().map(to_float).prefetch(AUTOTUNE)


def test_data(data_dir, batch_size, seed, jpeg_quality=None, subset=None):
    ds = limit(load_folder(os.path.join(data_dir, "test"), batch_size, seed), subset, batch_size)
    if jpeg_quality is not None:
        def jpeg(image, label):
            image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
            return tf.cast(tf.image.adjust_jpeg_quality(image, jpeg_quality), tf.float32), label
        ds = ds.unbatch().map(jpeg, num_parallel_calls=AUTOTUNE).batch(batch_size)
    return ds.prefetch(AUTOTUNE)


# ---------------------------------------------------------------- models

def build_model(backbone, img_size):
    inputs = keras.Input(shape=(IMG, IMG, 3))
    x = layers.RandomFlip("horizontal")(inputs)  # only active while training

    if backbone == "cnn":
        x = layers.Rescaling(1.0 / 255)(x)
        for filters in (32, 64, 128):
            # No BatchNormalization: on short CPU runs its running statistics made the model
            # score ~0.90 while training but ~0.59 on validation/test. Plain conv + dropout
            # behaves the same in training and testing.
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.25)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
    else:  # mobilenet
        x = layers.Resizing(img_size, img_size)(x)
        x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 expects pixels in [-1, 1]
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              input_shape=(img_size, img_size, 3), pooling="avg")
        x = layers.Dropout(0.2)(base(x))

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=f"cifake_{backbone}")


# ---------------------------------------------------------------- evaluation

def predict(model, ds):
    probs, labels = [], []
    for images, y in ds:
        probs.append(model.predict_on_batch(images).ravel())
        labels.append(y.numpy().ravel())
    return np.concatenate(labels).astype(int), np.concatenate(probs)


def summarise(y_true, prob_fake):
    y_pred = (prob_fake >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob_fake),
        "n_images": int(len(y_true)),
    }


def plot_confusion(y_true, prob_fake, path, title):
    cm = confusion_matrix(y_true, (prob_fake >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    p.add_argument("--backbone", choices=["cnn", "mobilenet"], default="cnn")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 for cnn, 1e-4 for mobilenet")
    p.add_argument("--img-size", type=int, default=96, help="MobileNetV2 input size: 96, 128, 160, 192 or 224")
    p.add_argument("--subset", type=int, default=None, help="train on only this many images")
    p.add_argument("--jpeg-qualities", type=int, nargs="*", default=[90, 70, 50])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            if not os.path.isdir(os.path.join(data_dir, split, cls)):
                raise SystemExit(f"Missing folder: {os.path.join(data_dir, split, cls)}")

    keras.utils.set_random_seed(args.seed)
    print("GPU:", tf.config.list_physical_devices("GPU") or "none found, using CPU")

    models_dir, results_dir = Path("models_tf"), Path("results_tf")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    name = args.backbone
    model_path = models_dir / f"best_{name}.keras"

    # Train
    train_ds, val_ds = train_val_data(data_dir, args.batch_size, args.subset, args.seed)
    model = build_model(args.backbone, args.img_size)
    lr = args.lr or (1e-3 if args.backbone == "cnn" else 1e-4)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
    ])
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (results_dir / f"history_{name}.json").write_text(json.dumps(hist, indent=2))
    print(f"Best validation accuracy: {max(hist['val_accuracy']):.4f}")

    # Evaluate on the untouched test set, then on JPEG-compressed copies
    model = keras.models.load_model(model_path)
    results = {}
    for quality in [None] + args.jpeg_qualities:
        ds = test_data(data_dir, 256, args.seed, jpeg_quality=quality)
        y_true, prob_fake = predict(model, ds)
        key = "original" if quality is None else f"jpeg_q{quality}"
        results[key] = summarise(y_true, prob_fake)
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[key].items()})
        if quality is None:
            plot_confusion(y_true, prob_fake, results_dir / f"confusion_matrix_{name}.png", f"Keras {name}")

    (results_dir / f"metrics_{name}.json").write_text(json.dumps(results, indent=2))
    print(f"Done. Model: {model_path} | results: {results_dir}/")


if __name__ == "__main__":
    main()
