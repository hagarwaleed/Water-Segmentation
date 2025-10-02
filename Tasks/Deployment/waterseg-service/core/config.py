import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CKPT = os.path.join(BASE_DIR, "checkpoints", "unet_resnet18_bce_dice_best.pth")

MODEL_CKPT_PATH = os.getenv("MODEL_CKPT_PATH", DEFAULT_CKPT)

THRESH = float(os.getenv("THRESH", "0.5"))

DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu"
