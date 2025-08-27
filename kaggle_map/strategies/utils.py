"""Utilities for strategy implementations."""

import torch
from loguru import logger


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug(
            "Device selection: using MPS (Apple Metal)",
            device_type="mps",
            available_backends=["mps", "cuda", "cpu"],
        )
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_count = torch.cuda.device_count()
        logger.debug(
            "Device selection: using CUDA",
            device_type="cuda",
            cuda_devices=cuda_count,
            available_backends=["cuda", "cpu"],
        )
        return device
    device = torch.device("cpu")
    logger.debug(
        "Device selection: fallback to CPU",
        device_type="cpu",
        available_backends=["cpu"],
    )
    return device
