"""Device selection utilities for PyTorch models."""

import torch
from loguru import logger


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug(
            "Device selection: using MPS (Apple Metal)",
            device_type="mps",
        )
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(
            "Device selection: using CUDA",
            device_type="cuda",
            cuda_devices=torch.cuda.device_count(),
        )
    else:
        device = torch.device("cpu")
        logger.debug(
            "Device selection: using CPU",
            device_type="cpu",
        )
    return device
