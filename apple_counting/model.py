"""Stage 2: ResNet50 count classifier (0-6 apples per crop)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7  # 0 (background) ... 6 apples
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def preprocess(image):
    """RGB uint8/float image -> 224x224, scaled to [0,1], ImageNet mean/std normalised."""
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def build_model():
    """ResNet50 (ImageNet) backbone + GAP -> 1024 -> 512 -> 7 softmax head."""
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(1024)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs=base.input, outputs=out), base


def load_model(path):
    return tf.keras.models.load_model(path)


def predict_count(model, crop_rgb):
    """Predict apple count for a single RGB crop. Returns (label, confidence)."""
    x = tf.expand_dims(preprocess(crop_rgb), 0)
    probs = model.predict(x, verbose=0)[0]
    label = int(np.argmax(probs))
    return label, float(probs[label])


def predict_counts(model, crops, batch_size=64):
    """Batched version of predict_count for a list of RGB crops."""
    if not crops:
        return [], []
    x = tf.stack([preprocess(c) for c in crops])
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    labels = probs.argmax(axis=1)
    return labels.tolist(), probs[np.arange(len(labels)), labels].tolist()
