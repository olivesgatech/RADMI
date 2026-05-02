"""
PyTorch Dataset for seismic sections.
"""

from typing import Callable, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset


class SectionLoader(Dataset):
    """
    Dataset yielding seismic sections and labels.
    
    Args:
        seismic: (N,H,W) or (N,1,H,W)
        labels: (N,H,W), optional
        indices: subset indices, optional
        transform: callable (section, label) -> (section, label)
    """

    def __init__(
        self,
        seismic: np.ndarray,
        labels: Optional[np.ndarray] = None,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        return_index: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.seismic = seismic
        self.labels = labels
        self.transform = transform
        self.return_index = return_index
        self.dtype = dtype

        if indices is None:
            self.indices = np.arange(seismic.shape[0])
        else:
            self.indices = np.asarray(indices)

        if labels is not None and labels.shape[0] != seismic.shape[0]:
            raise ValueError("Seismic and labels must have same first dimension.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])

        x = self.seismic[real_idx]
        if x.ndim == 2:
            x = x[None, :, :]
        elif x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(f"Unexpected section shape: {x.shape}")

        y = None
        if self.labels is not None:
            y = self.labels[real_idx]

        if self.transform is not None:
            x, y = self.transform(x, y)

        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dtype=self.dtype)
        
        if y is None:
            if self.return_index:
                return x_tensor, real_idx
            return x_tensor

        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))

        if self.return_index:
            return x_tensor, y_tensor, real_idx
        return x_tensor, y_tensor
