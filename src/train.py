"""Training loop, callbacks, and checkpointing for the CIFAR-10 CNN."""

import argparse
from pathlib import Path

import tensorflow as tf
import yaml

from src.data import load_datasets
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config["seed"]
    tf.keras.utils.set_random_seed(seed)

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    learning_rate = args.learning_rate or train_cfg["learning_rate"]

    train_ds, val_ds, _ = load_datasets(
        batch_size=batch_size,
        val_split=data_cfg["val_split"],
        augment=data_cfg["augment"],
        seed=seed,
    )

    model = build_model(
        input_shape=tuple(model_cfg["input_shape"]),
        num_classes=model_cfg["num_classes"],
        dropout=train_cfg["dropout"],
    )

    steps_per_epoch = train_ds.cardinality().numpy()
    decay_steps = max(steps_per_epoch, 1) * epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        alpha=train_cfg["lr_final"] / learning_rate,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
