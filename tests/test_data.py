"""Shape checks and augmentation-determinism checks for the data pipeline."""

import numpy as np
import tensorflow as tf

from src.data import augment_image


def test_augment_image_shape_and_dtype():
    image = tf.random.uniform([32, 32, 3], maxval=256, dtype=tf.int32)
    image = tf.cast(image, tf.uint8)

    augmented, label = augment_image(image, tf.constant(0))

    assert augmented.shape == (32, 32, 3)
    assert label.numpy() == 0


def test_augment_image_deterministic_under_fixed_seed():
    image = tf.random.uniform([32, 32, 3], maxval=256, dtype=tf.int32)
    image = tf.cast(image, tf.uint8)

    tf.keras.utils.set_random_seed(42)
    first = augment_image(image, tf.constant(0))[0].numpy()

    tf.keras.utils.set_random_seed(42)
    second = augment_image(image, tf.constant(0))[0].numpy()

    np.testing.assert_array_equal(first, second)
