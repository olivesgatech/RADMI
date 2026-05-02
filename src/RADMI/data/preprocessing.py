"""
Data loading and preprocessing for seismic sections.
"""

from typing import Optional, Tuple
import numpy as np


def load_data(
    path_seismic: str, 
    path_labels: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load seismic and labels from .npy files."""
    seismic = np.load(path_seismic)
    labels = np.load(path_labels) if path_labels is not None else None
    return seismic, labels


def standardize_features(
    seismic: np.ndarray,
    eps: float = 1e-8,
    per_section: bool = True,
) -> np.ndarray:
    """
    Z-score normalization.
    
    Args:
        seismic: (N,H,W) or (N,1,H,W)
        per_section: normalize each section independently
    """
    x = seismic.astype(np.float32, copy=False)

    if x.ndim == 4 and x.shape[1] == 1:
        if per_section:
            mean = x.mean(axis=(2, 3), keepdims=True)
            std = x.std(axis=(2, 3), keepdims=True)
        else:
            mean = x.mean(keepdims=True)
            std = x.std(keepdims=True)
        return (x - mean) / (std + eps)

    if x.ndim == 3:
        if per_section:
            mean = x.mean(axis=(1, 2), keepdims=True)
            std = x.std(axis=(1, 2), keepdims=True)
        else:
            mean = x.mean(keepdims=True)
            std = x.std(keepdims=True)
        return (x - mean) / (std + eps)

    raise ValueError(f"Unsupported shape: {x.shape}. Expected (N,H,W) or (N,1,H,W).")
