"""One-batch overfit check: catches a silently broken training step."""

import numpy as np
import tensorflow as tf

from src.model import build_model


def test_model_output_shape():
    model = build_model(input_shape=(32, 32, 3), num_classes=10)
    batch = tf.random.uniform([4, 32, 32, 3], maxval=256, dtype=tf.int32)
    batch = tf.cast(batch, tf.float32)

    output = model(batch)

    assert output.shape == (4, 10)


def test_model_can_overfit_single_batch():
    tf.keras.utils.set_random_seed(42)

    model = build_model(input_shape=(32, 32, 3), num_classes=10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x = np.random.randint(0, 256, size=(8, 32, 32, 3)).astype("float32")
    y = np.random.randint(0, 10, size=(8,))

    history = model.fit(x, y, epochs=25, verbose=0)

    assert history.history["accuracy"][-1] > history.history["accuracy"][0]
