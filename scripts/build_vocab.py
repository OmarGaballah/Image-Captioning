"""
Build and save the vocabulary from the COCO training captions.

Usage:
    python -m scripts.build_vocab \
        --annotation data/coco/annotations/captions_train2017.json \
        --output data/vocab.json \
        --min_freq 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.vocabulary import Vocabulary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, help="Path to captions_train2017.json")
    parser.add_argument("--output", required=True, help="Where to save vocab.json")
    parser.add_argument("--min_freq", type=int, default=5, help="Minimum word frequency")
    args = parser.parse_args()

    with open(args.annotation) as f:
        data = json.load(f)

    captions = [ann["caption"] for ann in data["annotations"]]
    print(f"Building vocabulary from {len(captions):,} captions...")

    vocab = Vocabulary()
    vocab.build(captions, min_freq=args.min_freq)
    vocab.save(args.output)

    print(f"Vocabulary size: {len(vocab):,} tokens  (min_freq={args.min_freq})")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
