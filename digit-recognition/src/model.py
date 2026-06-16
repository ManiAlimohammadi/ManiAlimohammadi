"""
CNN architecture for handwritten digit recognition.
Achieves ~99.4% accuracy on MNIST test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for digit recognition.

    Architecture:
        Conv2d(1→32, 3x3) → BN → ReLU → Conv2d(32→32, 3x3) → BN → ReLU → MaxPool → Dropout(0.25)
        Conv2d(32→64, 3x3) → BN → ReLU → Conv2d(64→64, 3x3) → BN → ReLU → MaxPool → Dropout(0.25)
        FC(1024→512) → BN → ReLU → Dropout(0.5) → FC(512→10)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(num_classes: int = 10, dropout_rate: float = 0.25) -> DigitCNN:
    return DigitCNN(num_classes=num_classes, dropout_rate=dropout_rate)
