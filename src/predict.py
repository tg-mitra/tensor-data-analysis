"""Single-image inference against a trained checkpoint."""

import argparse

import numpy as np
import tensorflow as tf

from src.data import CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.keras")
    return parser.parse_args()


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [32, 32])
    return tf.expand_dims(image, axis=0)


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.checkpoint)
    image = load_image(args.image)

    probs = model.predict(image, verbose=0)[0]
    class_id = int(np.argmax(probs))

    print(f"{CLASS_NAMES[class_id]} ({probs[class_id]:.2f})")


if __name__ == "__main__":
    main()
