"""
Training script for Google Colab.

Run these cells in order in your Colab notebook:

--- Cell 1: Mount Drive and install dependencies ---
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone <your-repo-url> image-caption  # or upload the project manually
    %cd image-caption
    !pip install -r requirements.txt
    import nltk; nltk.download('punkt')

--- Cell 2: Build vocabulary (only needed once) ---
    !python -m scripts.build_vocab \
        --annotation /content/drive/MyDrive/coco/annotations/captions_train2017.json \
        --output /content/drive/MyDrive/coco/vocab.json

--- Cell 3: Start training ---
    !python -m scripts.train_colab

--- Cell 4: Resume after a disconnect ---
    !python -m scripts.train_colab --resume /content/drive/MyDrive/coco/checkpoints/epoch_07.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train

DRIVE_BASE   = "/content/drive/MyDrive/coco"
LOCAL_BASE   = "/content/coco"           # local SSD — ~80x faster than Drive
CKPT_DIR     = f"{DRIVE_BASE}/checkpoints"  # checkpoints always saved to Drive

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None,
                    help="Path to a saved epoch_XX.pt checkpoint to resume from")
parser.add_argument("--num_workers", default=2, type=int,
                    help="DataLoader worker processes")
parser.add_argument("--local", action="store_true",
                    help="Read images from local /content/coco instead of Drive")
args = parser.parse_args()

data_dir  = LOCAL_BASE if args.local else DRIVE_BASE
vocab_path = f"{data_dir}/vocab.json"

train(
    data_dir=data_dir,
    vocab_path=vocab_path,
    checkpoint_dir=CKPT_DIR,
    epochs=20,
    batch_size=32,
    resume=args.resume,
    num_workers=args.num_workers,
)
