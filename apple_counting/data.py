"""Dataset loading for the MinneApple counting split and Dataset 2 counting subset."""
import glob
import json
import os
from pathlib import Path

import tensorflow as tf

from .model import preprocess


def read_csv_labels(folder, gt_name):
    """Read `<image_name>,<count>` lines; images live in `<folder>/images`."""
    paths, labels = [], []
    with open(os.path.join(folder, gt_name)) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            p = os.path.join(folder, "images", parts[0])
            if os.path.exists(p):
                paths.append(p)
                labels.append(int(parts[1]))
    return paths, labels


def read_json_labels(folder, gt_name="ground_truth.json"):
    """Read `{image_name: count}` JSON; images live in `<folder>/images`."""
    with open(os.path.join(folder, gt_name)) as f:
        gt = json.load(f)
    paths, labels = [], []
    for name, count in gt.items():
        p = os.path.join(folder, "images", name)
        if os.path.exists(p):
            paths.append(p)
            labels.append(int(count))
    return paths, labels


def load_split(folder):
    """Auto-detect the label file format used in each split folder."""
    for name in ("train_ground_truth.txt", "val_ground_truth.txt", "labels.txt"):
        if os.path.exists(os.path.join(folder, name)):
            return read_csv_labels(folder, name)
    if os.path.exists(os.path.join(folder, "ground_truth.json")):
        return read_json_labels(folder)
    raise FileNotFoundError(f"No label file found in {folder}")


def yolo_counts(split_dir):
    """Ground-truth apple count per image from YOLO label files (one box per line)."""
    mapping = {}
    for txt in glob.glob(os.path.join(split_dir, "labels", "*.txt")):
        with open(txt) as f:
            n = sum(1 for line in f if line.strip())
        mapping[os.path.join(split_dir, "images", Path(txt).stem + ".jpg")] = n
    return mapping


def _load(path, label):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    return preprocess(img), label


def make_dataset(paths, labels, batch_size=128, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )
