"""Evaluation helpers: per-class metrics table and confusion matrix plot."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_report(y_true, y_pred, name="set", num_classes=7):
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    print(f"\nMetrics: {name}")
    print(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<8}")
    print("-" * 52)
    for i in range(num_classes):
        print(f"{i:<8}{p[i]:<12.4f}{r[i]:<12.4f}{f1[i]:<12.4f}{s[i]:<8}")
    w = s if s.sum() else None
    summary = {
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=w)),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
    }
    print(
        f"\nMacro F1: {summary['macro_f1']:.4f} | Weighted F1: {summary['weighted_f1']:.4f}"
        f" | Accuracy: {summary['accuracy']:.4f}"
    )
    return summary


def plot_confusion_matrix(y_true, y_pred, name="set", out_path=None, num_classes=7):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
