"""Train the ResNet50 apple-count classifier on the MinneApple counting split.

Phase 1: backbone frozen, train the head (Adam 1e-3).
Phase 2: unfreeze everything and fine-tune (Adam 1e-4), keep best val accuracy.

Example:
    python train.py --data data/dataset_1/counting --out weights
"""
import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from apple_counting.data import load_split, make_dataset
from apple_counting.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing train/ and val/")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr_p, tr_y = load_split(os.path.join(args.data, "train"))
    va_p, va_y = load_split(os.path.join(args.data, "val"))
    print(f"train: {len(tr_p)} images | val: {len(va_p)} images")
    train_ds = make_dataset(tr_p, tr_y, args.batch_size, shuffle=True)
    val_ds = make_dataset(va_p, va_y, args.batch_size)

    model, base = build_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Phase 1: head only
    base.trainable = False
    model.compile(optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    model.fit(
        train_ds,
        epochs=args.head_epochs,
        validation_data=val_ds,
        callbacks=[
            callbacks.ModelCheckpoint(
                os.path.join(args.out, "best_head.h5"),
                save_best_only=True, monitor="val_accuracy", mode="max",
            )
        ],
    )

    # Phase 2: full fine-tune
    base.trainable = True
    model.compile(optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
    hist = model.fit(
        train_ds,
        epochs=args.finetune_epochs,
        validation_data=val_ds,
        callbacks=[
            callbacks.ModelCheckpoint(
                os.path.join(args.out, "best_finetune.h5"),
                save_best_only=True, monitor="val_accuracy", mode="max",
            ),
            callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ],
    )

    # Learning curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k in ("accuracy", "val_accuracy"):
        ax[0].plot(hist.history[k], label=k)
    for k in ("loss", "val_loss"):
        ax[1].plot(hist.history[k], label=k)
    for a, t in zip(ax, ("Accuracy", "Loss")):
        a.set_xlabel("Epoch"); a.set_title(t); a.legend()
    fig.savefig(os.path.join(args.out, "learning_curves.png"), dpi=150, bbox_inches="tight")
    print("Best fine-tuned weights:", os.path.join(args.out, "best_finetune.h5"))


if __name__ == "__main__":
    main()
