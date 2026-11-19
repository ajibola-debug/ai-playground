print("Training...")

"""Train MobileNetV2 on CIFAKE in two phases (the standard transfer-learning recipe).

Phase 1  The pretrained MobileNetV2 backbone is frozen; only the new classifier head learns.
         Fast, and stops the random head from wrecking the pretrained features.
Phase 2  The backbone is unfrozen and the whole network is fine-tuned with a small learning rate.

The backbone always runs with training=False, so its BatchNormalization layers keep their
ImageNet statistics. That avoids the train/test mismatch that broke the first small CNN.

Saves exactly what app_tensorflow.py looks for:
    models_tf/best_mobilenet.keras
    results_tf/metrics_mobilenet.json, confusion_matrix_mobilenet.png, history_mobilenet.json

Run from the project folder (needs cifake_tensorflow.py next to it):
    python train_mobilenet.py --data-dir cifake
    python train_mobilenet.py --data-dir cifake --subset 20000      # faster on a laptop CPU

On a CPU this is slow; Google Colab's free GPU is much faster.
"""
import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
import keras
from keras import layers

# Reuse the data loading and evaluation from the main script, so both models are tested identically.
from cifake_tensorflow import (CLASS_NAMES, IMG, plot_confusion, predict, summarise, test_data,
                               train_val_data)


def build_mobilenet(img_size):
    inputs = keras.Input(shape=(IMG, IMG, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.Resizing(img_size, img_size)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 expects pixels in [-1, 1]
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(img_size, img_size, 3), pooling="avg")
    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="cifake_mobilenet"), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    p.add_argument("--img-size", type=int, default=96, help="96, 128, 160, 192 or 224 (bigger = slower)")
    p.add_argument("--head-epochs", type=int, default=3, help="phase 1: backbone frozen")
    p.add_argument("--fine-epochs", type=int, default=5, help="phase 2: whole network fine-tuned")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--fine-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
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
    print("GPU:", tf.config.list_physical_devices("GPU") or "none found, using CPU (this will be slow)")

    models_dir, results_dir = Path("models_tf"), Path("results_tf")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_mobilenet.keras"

    train_ds, val_ds = train_val_data(data_dir, args.batch_size, args.subset, args.seed)
    model, base = build_mobilenet(args.img_size)

    def checkpoint(threshold=None):
        return keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", mode="max",
                                               save_best_only=True, initial_value_threshold=threshold)

    def early_stop():
        return keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2,
                                             restore_best_weights=True)

    # Phase 1: train the head only
    print("\n=== Phase 1: training the classifier head (backbone frozen) ===")
    base.trainable = False
    compile_model(model, args.head_lr)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                   callbacks=[checkpoint(), early_stop()])
    best_phase1 = max(h1.history["val_accuracy"])

    # Phase 2: fine-tune everything (recompile after changing trainable)
    print("\n=== Phase 2: fine-tuning the whole network ===")
    base.trainable = True
    compile_model(model, args.fine_lr)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_epochs,
                   callbacks=[checkpoint(best_phase1), early_stop()])

    history = {k: [float(v) for v in h1.history[k] + h2.history.get(k, [])] for k in h1.history}
    history["phase"] = [1] * len(h1.history["loss"]) + [2] * len(h2.history["loss"])
    (results_dir / "history_mobilenet.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest validation accuracy: {max(history['val_accuracy']):.4f}")

    # Evaluate the best checkpoint on the untouched test set, then on JPEG-compressed copies
    model = keras.models.load_model(model_path)
    results = {}
    for quality in [None] + args.jpeg_qualities:
        ds = test_data(data_dir, 256, args.seed, jpeg_quality=quality)
        y_true, prob_fake = predict(model, ds)
        key = "original" if quality is None else f"jpeg_q{quality}"
        results[key] = summarise(y_true, prob_fake)
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[key].items()})
        if quality is None:
            plot_confusion(y_true, prob_fake, results_dir / "confusion_matrix_mobilenet.png", "MobileNetV2")

    (results_dir / "metrics_mobilenet.json").write_text(json.dumps(results, indent=2))
    print(f"Done. Model: {model_path} | results: {results_dir}/ | the app will now show MobileNetV2.")


if __name__ == "__main__":
    main()
