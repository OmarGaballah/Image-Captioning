# YOLO Encoder: This module is responsible for image understanding.
# A pre-trained YOLOv8 model is used as the encoder. It extracts features from the image.
# The features are then projected to a lower dimension (d_model) and passed through a Transformer.


import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOFeatureExtractor(nn.Module):
    # YOLOv8 backbone is layers 0-9 (ends with SPPF); neck and head start at 10.
    # These layers are purely sequential — no skip connections — so wrapping them in nn.Sequential is safe.
    _BACKBONE_END = 10

    # model_name: "yolov8n.pt", "yolov8m.pt", etc.
    # d_model: The dimension of the model's output.
    # freeze: If True, freeze the weights of the backbone.
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        d_model: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        # Load pre-trained YOLOv8 model from ultralytics
        yolo = YOLO(model_name)
        
        # Extract the backbone layers (up to SPPF module)
        self.backbone = nn.Sequential(*list(yolo.model.model[: self._BACKBONE_END]))

        # Freeze backbone weights if specified
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Probe the backbone once to learn its output shape without hardcoding it.
        # This means the same class works for yolov8n, yolov8m, etc.
        with torch.no_grad():
            probe = self.backbone(torch.zeros(1, 3, 640, 640))  # [1, C, H, W]
            _, C, H, W = probe.shape

        # Linear layer: maps backbone channel dim → Transformer d_model
        self.proj = nn.Linear(C, d_model)

        # One learnable embedding per spatial location (400 for 640×640 input)
        self.pos_embed = nn.Parameter(torch.randn(1, H * W, d_model) * 0.02)

    # B: Batch size
    # C: Number of channels in the backbone output
    # H, W: Height and width of the feature map
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                 # [B, C, H, W]
        feat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        feat = self.proj(feat)                  # [B, H*W, d_model]
        return feat + self.pos_embed            # [B, H*W, d_model]
