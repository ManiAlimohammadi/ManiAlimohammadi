"""Unit tests for model architecture and forward pass."""

import torch
import pytest
from src.model import DigitCNN, get_model


def test_output_shape():
    model = get_model()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)


def test_single_sample():
    model = get_model()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_output_not_nan():
    model = get_model()
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


def test_parameter_count():
    model = get_model()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Should be between 500K and 5M
    assert 500_000 < n < 5_000_000


def test_dropout_rate():
    model = DigitCNN(dropout_rate=0.5)
    assert model is not None
