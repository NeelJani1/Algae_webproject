import torch
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

logger = logging.getLogger("SeaDino")

@dataclass
class SeaDinoConfig:
    """Stores all static configurations and magic numbers for the pipeline."""
    eval_h: int = 1088
    eval_w: int = 1920
    patch_size: int = 16
    dpi: int = 200
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    
    color_palette: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 0.0],  [0.8, 0.2, 0.2],  [0.2, 0.8, 0.2],  
        [0.8, 0.8, 0.2],  [0.2, 0.2, 0.8],  [0.8, 0.2, 0.8]   
    ]))

    def __post_init__(self):
        """Safety checks."""
        if self.eval_h % self.patch_size != 0:
            new_h = (self.eval_h // self.patch_size) * self.patch_size
            self.eval_h = new_h
            
        if self.eval_w % self.patch_size != 0:
            new_w = (self.eval_w // self.patch_size) * self.patch_size
            self.eval_w = new_w
            
        # Restrict DPI to a safe 100-600 range
        if not (100 <= self.dpi <= 600):
            logger.warning(f"DPI {self.dpi} is outside safe range (100-600). Clamping it.")
            self.dpi = max(100, min(self.dpi, 600))

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.eval_h // self.patch_size, self.eval_w // self.patch_size