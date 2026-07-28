import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DenseLinearProbeTiny(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.classifier(x)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

class SpatialDecoderProbeSmall(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.decoder(x), size=target_size, mode='bilinear', align_corners=False)

class SpatialDecoderProbeMedium(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    def forward(self, x, target_size):
        return F.interpolate(self.decoder(x), size=target_size, mode='bilinear', align_corners=False)

class SpatialDecoderProbeBig(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    def forward(self, x, target_size):
        return F.interpolate(self.decoder(x), size=target_size, mode='bilinear', align_corners=False)

# Mapping of size names to their respective PyTorch classes
PROBE_REGISTRY = {
    'tiny': DenseLinearProbeTiny, 
    'small': SpatialDecoderProbeSmall, 
    'medium': SpatialDecoderProbeMedium, 
    'big': SpatialDecoderProbeBig
}