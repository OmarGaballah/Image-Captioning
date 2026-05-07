# Main training entry point
# This script trains the image captioning model using the COCO dataset.
# It uses a pre-trained YOLOv8 model as the encoder and a Transformer-based decoder.
# The model is trained using teacher forcing with a cosine annealing learning rate scheduler.

import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.dataloader import get_dataloader
from src.data.dataset import COCOCaptionDataset
from src.data.vocabulary import Vocabulary
from src.model.captioner import ImageCaptioner
from src.model.decoder import CaptionDecoder
from src.model.encoder import YOLOFeatureExtractor


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------

# The learning rate scheduler. It starts with a warm-up period where the learning rate increases linearly,
# followed by a cosine annealing period where the learning rate decreases following a cosine curve.
# This helps the model to converge faster and achieve better performance.
# 
#  optimizer: The optimizer to use.
#  warmup_steps: The number of warm-up steps.
#  total_steps: The total number of steps.
#  Returns: A learning rate scheduler.

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


# ----------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    epoch: int,
    model: ImageCaptioner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    best_val_loss: float,
) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def _load_checkpoint(
    path: str,
    model: ImageCaptioner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Resumed from epoch {ckpt['epoch']} (best val loss so far: {ckpt['best_val_loss']:.4f})")
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


# ----------------------------------------------------------------------
# One epoch
# ----------------------------------------------------------------------

# Trains the model for one epoch.
# 
#  model: The model to train.
#  loader: The data loader.
#  optimizer: The optimizer.
#  scheduler: The learning rate scheduler.
#  criterion: The loss function.
#  device: The device to use.
#  Returns: The average loss for the epoch.

def _train_one_epoch(
    model: ImageCaptioner,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, captions, _ in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        captions = captions.to(device)

        # Teacher forcing:
        #   inp  = [<start>, w1, w2, ..., wN]   (drop last token)
        #   tgt  = [w1,      w2, ..., wN, <end>] (drop first token)
        # At each position t, the model sees the correct prefix and predicts t+1.
        inp = captions[:, :-1]
        tgt = captions[:, 1:]
        pad_mask = inp == Vocabulary.PAD_IDX

        logits = model(images, inp, pad_mask)   # [B, T, vocab_size]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # [B*T, vocab_size]
            tgt.reshape(-1),                      # [B*T]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Validates the model on the validation set.
# 
#  model: The model to validate.
#  loader: The data loader.
#  criterion: The loss function.
#  device: The device to use.
#  Returns: The average loss for the validation set.
@torch.no_grad()
def _validate(
    model: ImageCaptioner,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for images, captions, _ in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        captions = captions.to(device)

        inp = captions[:, :-1]
        tgt = captions[:, 1:]
        pad_mask = inp == Vocabulary.PAD_IDX

        logits = model(images, inp, pad_mask)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / len(loader)


# ----------------------------------------------------------------------
# Main training entry point
# ----------------------------------------------------------------------

# Trains the model for a specified number of epochs.
# 
#  data_dir: Directory containing the data.
#  vocab_path: Path to the vocabulary file.
#  checkpoint_dir: Directory to save checkpoints.
#  epochs: Number of epochs to train.
#  batch_size: Batch size.
#  lr: Learning rate.
#  d_model: Dimension of the model.
#  nhead: Number of attention heads.
#  num_layers: Number of layers.
#  warmup_steps: Number of warm-up steps.
#  num_workers: Number of worker processes.
#  Returns: None.
def train(
    data_dir: str,
    vocab_path: str,
    checkpoint_dir: str = "checkpoints",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 2,
    warmup_steps: int = 500,
    num_workers: int = 4,
    resume: str = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Data ---
    vocab = Vocabulary.load(vocab_path)

    train_ds = COCOCaptionDataset(
        image_dir=os.path.join(data_dir, "train2017"),
        annotation_file=os.path.join(data_dir, "annotations", "captions_train2017.json"),
        vocab=vocab,
        split="train",
    )
    val_ds = COCOCaptionDataset(
        image_dir=os.path.join(data_dir, "val2017"),
        annotation_file=os.path.join(data_dir, "annotations", "captions_val2017.json"),
        vocab=vocab,
        split="val",
    )
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = get_dataloader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Model ---
    encoder = YOLOFeatureExtractor(model_name="yolov8n.pt", d_model=d_model, freeze=True)
    decoder = CaptionDecoder(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )

    # ImageCaptioner: Combines the encoder and decoder.
    model = ImageCaptioner(encoder, decoder).to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # Calculate total steps for scheduler
    total_steps = epochs * len(train_loader)
    scheduler = _build_scheduler(optimizer, warmup_steps, total_steps)

    # ignore_index=PAD_IDX so padding positions don't contribute to the loss
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)

    # --- Resume ---
    start_epoch = 1
    best_val_loss = float("inf")

    if resume and os.path.exists(resume):
        start_epoch, best_val_loss = _load_checkpoint(resume, model, optimizer, scheduler, device)

    # --- Training loop ---
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss   = _validate(model, val_loader, criterion, device)
        print(f"  train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(os.path.join(checkpoint_dir, "best.pt"),
                             epoch, model, optimizer, scheduler, best_val_loss)
            print(f"  → best checkpoint saved (val loss: {val_loss:.4f})")

        # Always save latest epoch so Colab can resume after a disconnect
        _save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch:02d}.pt"),
                         epoch, model, optimizer, scheduler, best_val_loss)


if __name__ == "__main__":
    train(
        data_dir="data/coco",
        vocab_path="data/vocab.json",
    )
