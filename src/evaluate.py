"""Evaluation: metrics, confusion matrix, and error analysis for the trained model."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASS_NAMES, load_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.keras")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="artifacts/eval")
    return parser.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.checkpoint)
    _, _, test_ds = load_datasets(batch_size=args.batch_size, augment=False)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(report)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (output_dir / "classification_report.txt").write_text(report)

    misclassified = np.where(y_true != y_pred)[0]
    np.save(output_dir / "misclassified_indices.npy", misclassified[:50])


if __name__ == "__main__":
    main()
