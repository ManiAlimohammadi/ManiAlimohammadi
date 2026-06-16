"""
Evaluation utilities: per-class accuracy, confusion matrix, misclassification analysis.
"""

import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_detailed(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    n_classes = 10
    confusion = [[0] * n_classes for _ in range(n_classes)]
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

    overall_acc = sum(per_class_correct[c] for c in range(n_classes)) / len(all_labels)
    per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in range(n_classes)}

    return {
        "overall_accuracy": round(overall_acc, 6),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion,
        "total_samples": len(all_labels),
    }
