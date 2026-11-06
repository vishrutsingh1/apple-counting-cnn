"""Stage 1: colour-based apple region proposal.

SLIC superpixels in LAB space -> per-superpixel Gaussian -> KL divergence
against 15 apple / 15 background reference clusters -> binary mask ->
morphological opening -> rotated bounding boxes -> perspective-warped crops.
"""
import json
import os

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def equalise_L_lab(img_lab):
    """Histogram-equalise the L (lightness) channel to reduce lighting variation."""
    l, a, b = cv2.split(img_lab)
    return cv2.merge((cv2.equalizeHist(l), a, b))


def slic_superpixels_lab(img_lab, region_div=60, ruler=5.0, iterations=5):
    """SLICO superpixels on a LAB image. Returns (labels, n_superpixels)."""
    h, w, _ = img_lab.shape
    region_size = max(5, int(min(h, w) / region_div))
    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_lab, algorithm=cv2.ximgproc.SLICO, region_size=region_size, ruler=ruler
    )
    slic.iterate(iterations)
    slic.enforceLabelConnectivity(min_element_size=25)
    return slic.getLabels(), slic.getNumberOfSuperpixels()


def sp_mean_cov(pixels):
    """Mean and regularised covariance of the LAB pixels in one superpixel."""
    mu = pixels.mean(axis=0)
    X = pixels - mu
    sigma = (X.T @ X) / max(1, len(pixels) - 1)
    sigma += np.eye(3) * 1e-3
    return mu, sigma


def kl_gaussians(mu_q, sigma_q, mu_p, sigma_p):
    """KL( N(mu_q, sigma_q) || N(mu_p, sigma_p) ) for 3-D Gaussians."""
    inv_sp = np.linalg.inv(sigma_p)
    term1 = np.trace(inv_sp @ sigma_q)
    diff = mu_p - mu_q
    term2 = diff.T @ inv_sp @ diff
    term3 = np.log(np.linalg.det(sigma_p) / np.linalg.det(sigma_q) + 1e-12)
    return 0.5 * (term1 + term2 - 3 + term3)


def pad_rotated_rect(rr, pad, img_shape):
    """Grow a rotated rect by `pad` px per side, keeping its centre inside the image."""
    (cx, cy), (w, h), angle = rr
    H, W = img_shape[:2]
    cx = np.clip(cx, 0, W - 1)
    cy = np.clip(cy, 0, H - 1)
    return ((cx, cy), (w + 2 * pad, h + 2 * pad), angle)


def load_clusters(cluster_json_path):
    with open(cluster_json_path) as f:
        params = json.load(f)
    apple = [(np.array(p["mu"]), np.array(p["Sigma"])) for p in params if p["label"] == "apple"]
    bg = [(np.array(p["mu"]), np.array(p["Sigma"])) for p in params if p["label"] == "background"]
    return apple, bg


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def detect_and_crop_apples(
    img_bgr,
    cluster_json_path="assets/color_clusters.json",
    margin=3,
    pad=5,
    min_area=150,
    save_folder=None,
    base_name="image",
):
    """Find apple regions in a BGR image and return crops for the CNN counter.

    Returns a dict with keys: cropped, binary_mask, labels, img_rgb, contours, boxes_all.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    img_lab_eq = equalise_L_lab(cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB))

    apple_clusters, bg_clusters = load_clusters(cluster_json_path)
    labels, num_sp = slic_superpixels_lab(img_lab_eq)

    # Classify each superpixel: apple if closer (by `margin`) to an apple cluster
    binary_mask = np.zeros(labels.shape, dtype=np.uint8)
    for sp in range(num_sp):
        sp_mask = labels == sp
        if not np.any(sp_mask):
            continue
        mu, sigma = sp_mean_cov(img_lab_eq[sp_mask])
        kl_app = min(kl_gaussians(mu, sigma, m, s) for m, s in apple_clusters)
        kl_bg = min(kl_gaussians(mu, sigma, m, s) for m, s in bg_clusters)
        if kl_app + margin < kl_bg:
            binary_mask[sp_mask] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_images, boxes_all = [], []
    idx = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = pad_rotated_rect(cv2.minAreaRect(cnt), pad, img_rgb.shape)
        box = cv2.boxPoints(rect).astype(np.float32)
        boxes_all.append(box.astype(np.int32))

        w, h = map(int, rect[1])
        if w <= 0 or h <= 0:
            continue
        dst = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(
            img_rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        cropped_images.append(crop)

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_folder, f"{base_name}_apple_{idx}.png"),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            )
        idx += 1

    return {
        "cropped": cropped_images,
        "binary_mask": clean_mask,
        "labels": labels,
        "img_rgb": img_rgb,
        "contours": contours,
        "boxes_all": boxes_all,
    }


def mean_colour_superpixels(img_rgb, labels):
    """Paint every superpixel with its mean colour (for visualisation)."""
    out = np.zeros_like(img_rgb)
    for sp in range(labels.max() + 1):
        m = labels == sp
        if np.any(m):
            out[m] = img_rgb[m].mean(axis=0).astype(np.uint8)
    return out


def draw_boxes(img_rgb, boxes, colour=(0, 255, 0), thickness=2):
    out = img_rgb.copy()
    for box in boxes:
        cv2.drawContours(out, [box], 0, colour, thickness)
    return out
