import torch
import segmentation_models_pytorch as smp
from typing import Tuple
from core.config import MODEL_CKPT_PATH, DEVICE

class UNet13Inference:
    def __init__(self, ckpt_path: str = MODEL_CKPT_PATH, device: str = DEVICE):
        self.device = torch.device(device)
       
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,   
            in_channels=13,
            classes=1
        ).to(self.device)
        self.model.eval()
        self._load_weights(ckpt_path)

    def _load_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
    
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=True)

    @torch.no_grad()
    def forward_logits(self, chw: torch.Tensor) -> torch.Tensor:
        """
        chw: torch tensor (1,13,H,W) on self.device
        returns logits (1,1,H,W)
        """
        return self.model(chw)

    @torch.no_grad()
    def predict_logits_from_np(self, chw_np) -> torch.Tensor:
        x = torch.from_numpy(chw_np).unsqueeze(0).to(self.device, dtype=torch.float32)  # (1,13,H,W)
        return self.forward_logits(x)
