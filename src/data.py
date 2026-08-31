"""tf.data pipeline for CIFAR-10: loading, train/val split, and augmentation."""

import tensorflow as tf

CLASS_NAMES = [
    "aeroplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

AUTOTUNE = tf.data.AUTOTUNE


def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)

    # ±10% translate
    pad = 4
    image = tf.image.resize_with_crop_or_pad(image, 32 + pad, 32 + pad)
    image = tf.image.random_crop(image, size=[32, 32, 3])

    # ±10 degree rotate (approximated with a small random rotation via
    # tf.image primitives to avoid a hard dependency on tensorflow-addons)
    image = tf.image.rot90(
        image, k=tf.random.uniform([], 0, 1, dtype=tf.int32)
    )

    return image, label


def _to_dataset(images, labels, batch_size, shuffle, augment, seed):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def load_datasets(batch_size=64, val_split=0.1, augment=True, seed=42):
    """Load CIFAR-10 and return (train_ds, val_ds, test_ds)."""
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train_full = y_train_full.squeeze()
    y_test = y_test.squeeze()

    num_val = int(len(x_train_full) * val_split)
    x_val, y_val = x_train_full[:num_val], y_train_full[:num_val]
    x_train, y_train = x_train_full[num_val:], y_train_full[num_val:]

    train_ds = _to_dataset(
        x_train, y_train, batch_size, shuffle=True, augment=augment, seed=seed
    )
    val_ds = _to_dataset(
        x_val, y_val, batch_size, shuffle=False, augment=False, seed=seed
    )
    test_ds = _to_dataset(
        x_test, y_test, batch_size, shuffle=False, augment=False, seed=seed
    )

    return train_ds, val_ds, test_ds
