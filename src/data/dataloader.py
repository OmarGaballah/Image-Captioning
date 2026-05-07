# The DataLoader takes the Dataset and turns it into batches of data for the model.
# It handles shuffling (randomizing order) and multiprocessing (using multiple CPU cores to load images in parallel).
# Collate Function: The most critical part here is _collate_fn. Since captions have different lengths (e.g., "a dog" is 2 words, "a big brown dog" is 5), we can't just stack them into a tensor. We need to pad the shorter ones with <pad> tokens so they all match the length of the longest caption in the batch.
# This function is passed to the DataLoader, which uses it to construct each batch.

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data.dataset import COCOCaptionDataset
from src.data.vocabulary import Vocabulary


def _collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, captions = zip(*batch)

    # Lengths must be computed before padding
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)

    images = torch.stack(images, dim=0)                          # [B, C, H, W]
    captions = pad_sequence(captions, batch_first=True,          # [B, T]
                            padding_value=pad_idx)

    return images, captions, lengths


def get_dataloader(
    dataset: COCOCaptionDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: _collate_fn(b, pad_idx=Vocabulary.PAD_IDX),
        pin_memory=torch.cuda.is_available(),
    )
