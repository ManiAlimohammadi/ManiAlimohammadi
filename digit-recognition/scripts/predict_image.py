#!/usr/bin/env python3
"""
Predict the digit in one or more image files.

Usage:
    python scripts/predict_image.py path/to/image.png
    python scripts/predict_image.py img1.png img2.jpg --checkpoint models/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import load_model, predict


def main():
    p = argparse.ArgumentParser(description="Predict digits from image files")
    p.add_argument("images", nargs="+", help="Path(s) to image files")
    p.add_argument("--checkpoint", default="./models/best_model.pth")
    p.add_argument("--json", action="store_true", help="Output JSON")
    args = p.parse_args()

    model = load_model(args.checkpoint)

    results = []
    for img_path in args.images:
        result = predict(model, img_path)
        result["file"] = img_path
        results.append(result)

        if not args.json:
            print(f"{img_path}: digit={result['digit']}  confidence={result['confidence']:.2%}")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
