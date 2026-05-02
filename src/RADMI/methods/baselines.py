"""
Baseline uncertainty methods for comparison.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_ensemble_uncertainty(
    models: List[nn.Module],
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Compute predictive entropy from deep ensemble.
    
    Args:
        models: list of ensemble members
        x: (1, 1, H, W) input
        device: torch device
        
    Returns:
        (H, W) entropy map
    """
    x = x.to(device)
    all_probs = []

    with torch.no_grad():
        for m in models:
            logits, _ = m(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    all_probs = torch.stack(all_probs, dim=0)
    mean_probs = all_probs.mean(dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)

    return entropy.squeeze().cpu().numpy()


def compute_mc_dropout_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    n_forward: int = 20,
) -> np.ndarray:
    """
    Compute predictive entropy using MC-Dropout.
    
    Args:
        model: model with dropout layers
        x: (1, 1, H, W) input
        n_forward: number of stochastic forward passes
        
    Returns:
        (H, W) entropy map
    """
    x = x.to(device)
    model.train()  # enable dropout

    predictions = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits, _ = model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu())

    model.eval()

    predictions = torch.stack(predictions, dim=0)
    mean_probs = predictions.mean(dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)

    return entropy.squeeze().numpy()


def compute_softmax_entropy(logits: torch.Tensor) -> np.ndarray:
    """
    Compute entropy of softmax distribution.
    
    Args:
        logits: (1, C, H, W) model output
        
    Returns:
        (H, W) entropy map
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.squeeze().cpu().numpy()


def compute_msp(logits: torch.Tensor) -> np.ndarray:
    """
    Compute 1 - max softmax probability (higher = more uncertain).
    
    Args:
        logits: (1, C, H, W) model output
        
    Returns:
        (H, W) uncertainty map
    """
    probs = F.softmax(logits, dim=1)
    max_probs = torch.max(probs, dim=1)[0]
    return (1 - max_probs).squeeze().cpu().numpy()
