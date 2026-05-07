"""
Generate a caption for a single image using the trained model.

Usage:
    python -m scripts.caption --image path/to/image.jpg
    python -m scripts.caption --image path/to/image.jpg --method greedy
    python -m scripts.caption --image path/to/image.jpg --method beam --beam_size 5
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.vocabulary import Vocabulary
from src.model.captioner import ImageCaptioner
from src.model.decoder import CaptionDecoder
from src.model.encoder import YOLOFeatureExtractor

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def load_image(path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, 640, 640]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True,          help="Path to input image")
    parser.add_argument("--vocab",       default="data/vocab.json")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--method",      default="both",         choices=["greedy", "beam", "both"])
    parser.add_argument("--beam_size",   default=3, type=int)
    parser.add_argument("--d_model",     default=256, type=int)
    parser.add_argument("--nhead",       default=8, type=int)
    parser.add_argument("--num_layers",  default=2, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    vocab = Vocabulary.load(args.vocab)
    encoder = YOLOFeatureExtractor(model_name="yolov8n.pt", d_model=args.d_model, freeze=True)
    decoder = CaptionDecoder(vocab_size=len(vocab), d_model=args.d_model,
                             nhead=args.nhead, num_layers=args.num_layers)
    model = ImageCaptioner(encoder, decoder).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # --- Load image ---
    image = load_image(args.image).to(device)

    # --- Generate ---
    print(f"\nImage: {args.image}\n")

    if args.method in ("greedy", "both"):
        ids = model.caption(image, Vocabulary.START_IDX, Vocabulary.END_IDX, method="greedy")
        print(f"Greedy    : {vocab.decode(ids)}")

    if args.method in ("beam", "both"):
        ids = model.caption(image, Vocabulary.START_IDX, Vocabulary.END_IDX,
                            method="beam", beam_size=args.beam_size)
        print(f"Beam (k={args.beam_size}): {vocab.decode(ids)}")


if __name__ == "__main__":
    main()
