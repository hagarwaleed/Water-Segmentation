import io, base64
import numpy as np
from PIL import Image
import torch

def logits_to_mask_np(logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """
    logits: (1,1,H,W) or (1,H,W) torch tensor
    returns: (H,W) float32 {0,1}
    """
    if logits.ndim == 4:
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    elif logits.ndim == 3:
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    else:
        raise ValueError("Unexpected logits shape")
    return (probs >= thr).astype(np.float32)

def mask_to_png_b64(mask01: np.ndarray) -> str:
    """
    mask01: (H,W) in {0,1}
    returns: base64 PNG string
    """
    img = Image.fromarray((mask01 * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
