"""Full pipeline: segment apple regions, classify each crop's count, sum per image.

Single image (saves a visualisation):
    python count_apples.py --weights weights/best_finetune.h5 --image assets/demo/example_tree.jpg

Whole YOLO-labelled split (reports MAE against box counts):
    python count_apples.py --weights weights/best_finetune.h5 --yolo-split data/roboflow_data/test
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from apple_counting.data import yolo_counts
from apple_counting.model import load_model, predict_counts
from apple_counting.segmentation import (
    detect_and_crop_apples,
    draw_boxes,
    mean_colour_superpixels,
)


def count_image(model, img_bgr, clusters):
    res = detect_and_crop_apples(img_bgr, cluster_json_path=clusters)
    labels, conf = predict_counts(model, res["cropped"])
    return int(sum(labels)), res, labels


def visualise(res, labels, total, out_path):
    stages = [
        ("Original", res["img_rgb"], None),
        ("Mean-colour superpixels", mean_colour_superpixels(res["img_rgb"], res["labels"]), None),
        ("Apple mask", res["binary_mask"], "gray"),
        (f"Boxes - predicted total: {total}", draw_boxes(res["img_rgb"], res["boxes_all"]), None),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, (t, im, cmap) in zip(ax, stages):
        a.imshow(im, cmap=cmap); a.set_title(t); a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clusters", default="assets/color_clusters.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--yolo-split", help="folder with images/ and labels/ (YOLO format)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.weights)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(args.image)
        total, res, labels = count_image(model, img, args.clusters)
        print(f"{len(res['cropped'])} regions, per-region counts {labels}, total = {total}")
        stem = os.path.splitext(os.path.basename(args.image))[0]
        visualise(res, labels, total, os.path.join(args.out, f"{stem}_pipeline.png"))
        return

    gt = yolo_counts(args.yolo_split)
    errors, preds, trues = [], [], []
    for i, (path, true_n) in enumerate(sorted(gt.items()), 1):
        img = cv2.imread(path)
        pred_n = count_image(model, img, args.clusters)[0] if img is not None else 0
        errors.append(abs(pred_n - true_n)); preds.append(pred_n); trues.append(true_n)
        print(f"[{i}/{len(gt)}] pred {pred_n:3d} | true {true_n:3d} | {os.path.basename(path)}")

    errors = np.array(errors)
    print(f"\nImages: {len(errors)} | MAE: {errors.mean():.2f} | max error: {errors.max()}"
          f" | mean true count: {np.mean(trues):.1f} | mean predicted: {np.mean(preds):.1f}")

    plt.figure()
    bins = np.linspace(0, max(max(trues), max(preds)) + 1, 20)
    plt.hist(trues, bins=bins, alpha=0.6, label="ground truth")
    plt.hist(preds, bins=bins, alpha=0.6, label="predicted")
    plt.xlabel("Apples per image"); plt.ylabel("Images"); plt.legend()
    plt.savefig(os.path.join(args.out, "count_distribution.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
