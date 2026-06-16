"""Unit tests for prediction pipeline."""

import torch
import pytest
from PIL import Image
import numpy as np
from unittest.mock import patch
from src.model import get_model
from src.predict import predict


@pytest.fixture
def dummy_model():
    model = get_model()
    model.eval()
    return model


def test_predict_pil_image(dummy_model):
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    result = predict(dummy_model, img)
    assert "digit" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert 0 <= result["digit"] <= 9
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 10


def test_predict_sum_to_one(dummy_model):
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
    result = predict(dummy_model, img)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-3
