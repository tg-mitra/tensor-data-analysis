# tensor-data-analysis

A small, self-contained TensorFlow project that trains a convolutional neural network to classify images from the CIFAR-10 dataset. Built as a reference implementation — clean data pipeline, reproducible training loop, and a saved model you can actually load and serve

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Note:** This is a demonstration project. The metrics below are from an example run and are placeholders — regenerate them with `python -m src.evaluate` before quoting them anywhere.

---

## Overview

CIFAR-10 is 60,000 32×32 colour images across 10 classes — aeroplane, car, bird, cat, deer, dog, frog, horse, ship, truck. This repo trains a compact CNN on it end to end:

- **Input pipeline** built on `tf.data` with prefetching, shuffling and on-the-fly augmentation
- **Model** — four convolutional blocks with batch normalisation and dropout, roughly 550K parameters
- **Training** — Adam with cosine decay, early stopping on validation loss, checkpointing to `artifacts/`
- **Evaluation** — per-class precision/recall, confusion matrix, and a handful of misclassified samples written to disk for inspection

Small enough to train on a laptop CPU in under an hour; a few minutes on any modern GPU.

---

## Project structure

```
tensor-data-analysis/
├── src/
│   ├── data.py          # tf.data pipeline, augmentation, train/val split
│   ├── model.py         # CNN architecture definition
│   ├── train.py         # training loop, callbacks, checkpointing
│   ├── evaluate.py      # metrics, confusion matrix, error analysis
│   └── predict.py       # single-image inference
├── notebooks/
│   └── exploration.ipynb
├── artifacts/           # saved models and checkpoints (gitignored)
├── configs/
│   └── default.yaml     # hyperparameters
├── tests/
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/tg-mitra/tensor-data-analysis.git
cd tensor-data-analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train:

```bash
python -m src.train --config configs/default.yaml
```

Evaluate the trained checkpoint:

```bash
python -m src.evaluate --checkpoint artifacts/best_model.keras
```

Classify a single image:

```bash
python -m src.predict --image samples/cat.png
# → cat (0.94)
```

---

## Configuration

Hyperparameters live in `configs/default.yaml`; anything there can be overridden on the command line.

| Parameter | Default | Notes |
| --- | --- | --- |
| `batch_size` | 64 | Drop to 32 if you run out of GPU memory |
| `epochs` | 50 | Early stopping usually halts around 30–35 |
| `learning_rate` | 1e-3 | Cosine decay to 1e-5 |
| `dropout` | 0.3 | Applied after each conv block |
| `augment` | `true` | Random flip, ±10% translate, ±10° rotate |
| `seed` | 42 | Set for reproducible splits |

```bash
python -m src.train --epochs 30 --batch-size 32 --learning-rate 5e-4
```

---

## Model architecture

```
Input (32, 32, 3)
  └─ Rescaling(1./255)
  └─ [Conv2D(32) → BatchNorm → ReLU] ×2 → MaxPool → Dropout(0.3)
  └─ [Conv2D(64) → BatchNorm → ReLU] ×2 → MaxPool → Dropout(0.3)
  └─ [Conv2D(128) → BatchNorm → ReLU] ×2 → MaxPool → Dropout(0.4)
  └─ GlobalAveragePooling2D
  └─ Dense(128) → ReLU → Dropout(0.5)
  └─ Dense(10) → Softmax
```

Global average pooling instead of a flatten-and-dense head keeps the parameter count down and, in practice here, generalises a little better.

---

## Results

Example run, 35 epochs, single T4 GPU. **Placeholder figures — regenerate before citing.**

| Metric | Value |
| --- | --- |
| Test accuracy | 0.87 |
| Test loss | 0.41 |
| Macro F1 | 0.86 |
| Training time | ~8 min |

Weakest classes are cat and dog, which is the usual CIFAR-10 story — they account for a disproportionate share of the confusion matrix's off-diagonal mass. Most of the remaining error is those two plus bird/aeroplane.

---

## Testing

```bash
pytest tests/ -v
```

Covers the data pipeline shapes, the augmentation determinism under a fixed seed, and a one-batch overfit check that catches a silently broken training step.

---

## Roadmap

- [ ] Swap the custom CNN for a fine-tuned EfficientNet baseline
- [ ] Mixed-precision training
- [ ] Export to TFLite for on-device inference
- [ ] TensorBoard callback wired into the training loop
- [ ] Dockerfile for reproducible runs

---

## License

MIT — see [LICENSE](LICENSE).
