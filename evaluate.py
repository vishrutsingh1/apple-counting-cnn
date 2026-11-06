"""Evaluate the count classifier on labelled crops (confusion matrix + per-class F1).

Examples:
    python evaluate.py --weights weights/best_finetune.h5 --split data/dataset_1/counting/val
    python evaluate.py --weights weights/best_finetune.h5 --split data/dataset_2/d1 data/dataset_2/d2 --name "Dataset 2"
"""
import argparse
import os

import numpy as np

from apple_counting.data import load_split, make_dataset
from apple_counting.metrics import classification_report, plot_confusion_matrix
from apple_counting.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", nargs="+", required=True, help="one or more split folders (merged)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    paths, labels = [], []
    for s in args.split:
        p, y = load_split(s)
        paths += p
        labels += y
    name = args.name or os.path.basename(os.path.normpath(args.split[0]))
    print(f"{name}: {len(paths)} images")

    model = load_model(args.weights)
    preds = np.argmax(model.predict(make_dataset(paths, labels)), axis=1)

    os.makedirs(args.out, exist_ok=True)
    classification_report(labels, preds, name)
    plot_confusion_matrix(
        labels, preds, name, out_path=os.path.join(args.out, f"cm_{name.replace(' ', '_')}.png")
    )


if __name__ == "__main__":
    main()
