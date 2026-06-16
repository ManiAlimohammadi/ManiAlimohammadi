"""
Inference utilities: predict a digit from a file path or PIL image.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import get_model

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_model(checkpoint_path: str, device: torch.device | None = None) -> torch.nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: str | Path | Image.Image,
    device: torch.device | None = None,
) -> dict:
    """
    Returns a dict with keys: digit (int), confidence (float), probabilities (list[float]).
    """
    if device is None:
        device = next(model.parameters()).device

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("L")

    tensor = _TRANSFORM(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze()

    digit = probs.argmax().item()
    confidence = probs[digit].item()

    return {
        "digit": digit,
        "confidence": round(confidence, 4),
        "probabilities": {i: round(p.item(), 4) for i, p in enumerate(probs)},
    }
