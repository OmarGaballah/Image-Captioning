#  given an index, how do we get one (image, caption) pair ready for the model?

import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.vocabulary import Vocabulary

# ImageNet stats — safe to use since YOLO backbone was pretrained on ImageNet
# YOLO expects square input; 640 is the standard YOLOv8 resolution

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_INPUT_SIZE = 640

# RandomHorizontalFlip(): Randomly flips the image left-to-right.
# This doubles the effective dataset size by teaching the model that the right side is also the left.
# ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2): Randomly adjusts brightness, contrast, and color.
# This makes the model robust to different lighting conditions (e.g., sunny vs. cloudy days).
# transforms.ToTensor(): Converts the PIL image (H, W, C) to a PyTorch tensor (C, H, W) and scales values from [0, 255] to [0.0, 1.0].
# transforms.Normalize(mean=_MEAN, std=_STD): Normalizes the pixel values to match the distribution of the dataset the YOLO backbone was originally trained on (ImageNet).

def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

# During evaluation, we want consistency. We do NOT want the image to flip randomly, 
# because we need to compare the output to the exact same reference caption every time.
def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


class COCOCaptionDataset(Dataset):
    """
    Each item is one (image, caption) pair.
    During training a random caption is picked from the 5 available.
    During eval the first caption is used for consistent batching;
    call get_all_captions(image_id) for full BLEU evaluation.
    """

    # image_dir: Path to the folder containing images.
    # annotation_file: Path to the JSON file containing captions.
    # vocab: The Vocabulary object (built from the captions).
    # split: "train" or "val". Determines which transforms and captions to use.
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        vocab: Vocabulary,
        split: str = "train",
    ) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
        self.image_dir = image_dir
        self.vocab = vocab
        self.split = split
        self.transform = _train_transform() if split == "train" else _eval_transform()

        with open(annotation_file) as f:
            data = json.load(f)

        self._id_to_filename: dict[int, str] = {
            img["id"]: img["file_name"] for img in data["images"]
        }

        self._id_to_captions: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            self._id_to_captions.setdefault(img_id, []).append(ann["caption"])

        self.image_ids: list[int] = list(self._id_to_captions.keys())

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    # Returns the total number of images in the dataset.
    def __len__(self) -> int:
        return len(self.image_ids)

    # Gets the image and caption for a given index.
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        image = self._load_image(img_id)
        caption = self._pick_caption(img_id)
        tokens = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return image, tokens

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    # Returns all reference captions for an image as token lists.
    # BLEU score needs all 5 reference captions per image to compute correctly — one caption isn't enough.
    # But you can't put 5 variable-length caption lists into a standard batch. 
    # So during evaluation, you generate a caption with the model, then call this method to get all 5 references to score against.
    def get_all_captions(self, img_id: int) -> list[list[int]]:
        return [self.vocab.encode(c) for c in self._id_to_captions[img_id]]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Loads and preprocesses a single image.
    def _load_image(self, img_id: int) -> torch.Tensor:
        path = os.path.join(self.image_dir, self._id_to_filename[img_id])
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    # Picks a caption for an image.
    def _pick_caption(self, img_id: int) -> str:
        captions = self._id_to_captions[img_id]
        return random.choice(captions) if self.split == "train" else captions[0]
