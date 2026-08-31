"""CNN architecture for CIFAR-10 classification."""

from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters, dropout_rate):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(input_shape=(32, 32, 3), num_classes=10, dropout=0.3):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = conv_block(x, 32, dropout)
    x = conv_block(x, 64, dropout)
    x = conv_block(x, 128, dropout + 0.1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout + 0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cifar10_cnn")
