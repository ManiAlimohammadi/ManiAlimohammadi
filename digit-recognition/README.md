# Digit Recognition with Deep Learning

A production-ready CNN that recognises handwritten digits (0–9) from images, trained on MNIST.
Reaches **~99.4 % test accuracy** out of the box.

---

## Architecture

```
Input (1×28×28)
  │
  ├─ Conv Block 1 ── Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout
  ├─ Conv Block 2 ── Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout
  └─ Classifier ──── FC(3136→512) → BN → ReLU → Dropout → FC(512→10)
```

Key design choices:
- **Batch Normalization** after every conv layer for training stability
- **OneCycleLR** scheduler for fast convergence
- **Label smoothing** (ε = 0.1) to prevent overconfident predictions
- **Mixed precision** (AMP) when a GPU is available
- **Early stopping** with configurable patience

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/ManiAlimohammadi/digit-recognition.git
cd digit-recognition
pip install -r requirements.txt

# 2. Train (downloads MNIST automatically)
python scripts/run_training.py

# 3. Predict
python scripts/predict_image.py path/to/digit.png
```

---

## Training

```bash
python scripts/run_training.py \
  --epochs 30          \   # max epochs (early stopping kicks in)
  --batch-size 128     \
  --lr 3e-3            \
  --patience 10        \
  --checkpoint-dir ./models
```

Progress is logged to stdout; the best checkpoint is saved to `models/best_model.pth`.

---

## Inference

```python
from src.predict import load_model, predict

model = load_model("models/best_model.pth")
result = predict(model, "my_digit.png")

print(result["digit"])        # e.g. 7
print(result["confidence"])   # e.g. 0.9987
```

Or via CLI:
```bash
python scripts/predict_image.py img1.png img2.png --json
```

---

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **~99.4 %** |
| Parameters | ~1.2 M |
| Training time (CPU) | ~15 min |
| Training time (GPU) | ~3 min |

---

## Project Structure

```
digit-recognition/
├── src/
│   ├── model.py        # CNN architecture
│   ├── dataset.py      # MNIST loaders + augmentation
│   ├── train.py        # Training loop, early stopping, checkpointing
│   ├── evaluate.py     # Detailed evaluation + confusion matrix
│   └── predict.py      # Single-image inference
├── scripts/
│   ├── run_training.py # CLI training entry-point
│   └── predict_image.py# CLI inference entry-point
├── tests/
│   ├── test_model.py
│   └── test_predict.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── pyproject.toml
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## License

MIT © Mani Alimohammadi
