# The Image Captioner is the main model that combines the encoder and decoder.

import torch
import torch.nn as nn

from src.model.encoder import YOLOFeatureExtractor
from src.model.decoder import CaptionDecoder

# encoder: YOLOFeatureExtractor
# decoder: CaptionDecoder
class ImageCaptioner(nn.Module):
    def __init__(self, encoder: YOLOFeatureExtractor, decoder: CaptionDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # images: [B, 3, 640, 640]  — batch of images
    # captions: [B, T]  — batch of token indices
    # pad_mask: [B, T]  — True at <pad> positions
    def forward(
        self,
        images: torch.Tensor,           # [B, 3, 640, 640]
        captions: torch.Tensor,         # [B, T]
        pad_mask: torch.Tensor = None,  # [B, T]  True at <pad> positions
    ) -> torch.Tensor:                  # [B, T, vocab_size]
        visual_tokens = self.encoder(images)
        return self.decoder(captions, visual_tokens, pad_mask)

    # image: [1, 3, 640, 640]  — single image, batched
    # start_idx: Starting token index
    # end_idx: Ending token index
    # method: "greedy" or "beam"
    # **kwargs: Additional arguments for beam search (beam_size, max_len)
    @torch.no_grad()
    def caption(
        self,
        image: torch.Tensor,  # [1, 3, 640, 640]  — single image, batched
        start_idx: int,
        end_idx: int,
        method: str = "beam",
        **kwargs,
    ) -> list[int]:
        self.eval()
        visual_tokens = self.encoder(image)
        if method == "greedy":
            return self.decoder.generate_greedy(visual_tokens, start_idx, end_idx, **kwargs)
        return self.decoder.generate_beam(visual_tokens, start_idx, end_idx, **kwargs)
