#!/usr/bin/env python3
"""
Entry-point script to train the digit recognition model.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --epochs 30 --batch-size 256 --lr 3e-3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_loaders
from src.evaluate import evaluate_detailed
from src.model import get_model
from src.predict import load_model
from src.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train digit recognition CNN")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--checkpoint-dir", default="./models")
    p.add_argument("--dropout", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info(f"Config: {vars(args)}")

    train_loader, val_loader, test_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    logger.info(
        f"Dataset: train={len(train_loader.dataset):,} | "
        f"val={len(val_loader.dataset):,} | "
        f"test={len(test_loader.dataset):,}"
    )

    model = get_model(dropout_rate=args.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    result = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    logger.info(f"Best val accuracy: {result['best_val_acc']:.4f}")

    # Final evaluation on test set
    best_model = load_model(result["checkpoint"])
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)

    test_results = evaluate_detailed(best_model, test_loader, device=device)
    logger.info(f"Test accuracy: {test_results['overall_accuracy']:.4f}")
    logger.info("Per-class accuracy:")
    for digit, acc in test_results["per_class_accuracy"].items():
        logger.info(f"  Digit {digit}: {acc:.4f}")

    # Save results summary
    summary = {
        "best_val_accuracy": result["best_val_acc"],
        "test_accuracy": test_results["overall_accuracy"],
        "per_class_accuracy": test_results["per_class_accuracy"],
        "confusion_matrix": test_results["confusion_matrix"],
        "config": vars(args),
    }
    out_path = Path(args.checkpoint_dir) / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
