"""
apps/quick_test.py

Quick local test WITHOUT sdks.novavision, using open_clip directly.
For trying out CLIP right away on your own machine without a GPU;
the real Novavision platform flow is `apps/inference.py`.

Usage:
    pip install open_clip_torch pillow numpy torch
    python apps/quick_test.py --image resources/sample.jpg --version ViT-B-16
    python apps/quick_test.py --text "a red buoy on water" --version ViT-B-16
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import open_clip
    import torch
    from PIL import Image as PILImage

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--version", type=str, default="ViT-B-16",
                         choices=["ViT-B-16", "ViT-B-32", "RN50"])
    args = parser.parse_args()

    if not args.image and not args.text:
        parser.error("At least one of --image or --text must be given")

    # OpenAI weights were trained with QuickGELU; without force_quick_gelu=True
    # embedding quality silently degrades (see src/utils/utils.py::_load_clip)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.version, pretrained="openai", force_quick_gelu=True
    )
    model = model.eval()  # falls back to CPU automatically if no GPU
    tokenizer = open_clip.get_tokenizer(args.version)

    with torch.no_grad():
        if args.image:
            pil_image = PILImage.open(args.image).convert("RGB")
            tensor = preprocess(pil_image).unsqueeze(0)
            features = model.encode_image(tensor)
            input_type = "image"
        else:
            tokens = tokenizer([args.text])
            features = model.encode_text(tokens)
            input_type = "text"

        features = features / features.norm(dim=-1, keepdim=True)
        embedding = features.squeeze(0).numpy()

    result = {
        "model_version": args.version,
        "input_type": input_type,
        "embedding_dim": int(embedding.shape[0]),
        "embedding_preview": embedding[:8].tolist(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
