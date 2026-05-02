"""
RADMI: Resolution-Aggregated Decoder Mutual Information.

Computes uncertainty by measuring MI between consecutive decoder layers.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F

from .mi_gaussian import mutual_information_gaussian


def compute_layer_mi(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    patch_size: int = 4,
    stride: int = 1,
    max_channels: int = 100,
) -> torch.Tensor:
    """
    Compute MI map between two feature tensors using patches.
    
    Args:
        Z1: (1, C1, H1, W1) first layer features
        Z2: (1, C2, H2, W2) second layer features (will be resized to match Z1)
        patch_size: spatial patch size for MI estimation
        stride: stride for patch extraction
        max_channels: limit channels for stable estimation
        
    Returns:
        (H_patches, W_patches) MI map
    """
    C1, H, W = Z1.shape[1], Z1.shape[2], Z1.shape[3]
    C2 = Z2.shape[1]
    
    # resize Z2 to match Z1 spatial dims
    if Z2.shape[2] != H or Z2.shape[3] != W:
        Z2 = F.interpolate(Z2, size=(H, W), mode='bilinear', align_corners=False)
    
    C_min = min(C1, C2, max_channels)
    
    Z1 = Z1[0, :C_min, :, :]
    Z2 = Z2[0, :C_min, :, :]

    # extract patches
    p1 = Z1.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    p2 = Z2.unfold(1, patch_size, stride).unfold(2, patch_size, stride)

    H_patches = p1.shape[1]
    W_patches = p1.shape[2]
    N_patches = H_patches * W_patches

    # reshape to (N_patches, patch_size^2, C)
    p1 = p1.permute(1, 2, 3, 4, 0).reshape(N_patches, patch_size * patch_size, C_min)
    p2 = p2.permute(1, 2, 3, 4, 0).reshape(N_patches, patch_size * patch_size, C_min)

    mi_vals = mutual_information_gaussian(p1, p2)
    mi_vals = mi_vals / C_min  # normalize by channels
    mi_vals = torch.clamp(mi_vals, min=0)

    return mi_vals.view(H_patches, W_patches)


def compute_radmi(
    features: Dict[str, torch.Tensor],
    target_shape: Tuple[int, int],
    patch_size: int = 4,
    stride: int = 1,
    max_channels: int = 100,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute RADMI uncertainty map from decoder features.
    
    Uses resolution-weighted combination of MI between adjacent decoder layers.
    
    Args:
        features: dict with keys 'up1', 'up2', 'up3', 'up4' containing feature tensors
        target_shape: (H, W) output size
        patch_size: patch size for MI computation
        stride: stride for patch extraction
        max_channels: max channels to use
        device: torch device
        
    Returns:
        (H, W) uncertainty map
    """
    if device is None:
        device = features["up1"].device

    up1 = features["up1"]
    up2 = features["up2"]
    up3 = features["up3"]
    up4 = features["up4"]

    # compute MI between adjacent decoder layers
    # up1 <-> up2
    up2_down = F.interpolate(up2, size=(up1.shape[2], up1.shape[3]), 
                              mode='bilinear', align_corners=False)
    mi_12 = compute_layer_mi(up1, up2_down, patch_size, stride, max_channels)

    # up2 <-> up3
    up3_down = F.interpolate(up3, size=(up2.shape[2], up2.shape[3]), 
                              mode='bilinear', align_corners=False)
    mi_23 = compute_layer_mi(up2, up3_down, patch_size, stride, max_channels)

    # up3 <-> up4
    up4_down = F.interpolate(up4, size=(up3.shape[2], up3.shape[3]), 
                              mode='bilinear', align_corners=False)
    mi_34 = compute_layer_mi(up3, up4_down, patch_size, stride, max_channels)

    # resolution-weighted combination
    mi_maps = [mi_12, mi_23, mi_34]
    resolutions = [m.shape[0] * m.shape[1] for m in mi_maps]
    total_res = sum(resolutions)
    weights = [r / total_res for r in resolutions]

    combined = torch.zeros(target_shape, device=device)
    for mi_map, weight in zip(mi_maps, weights):
        mi_up = F.interpolate(
            mi_map.unsqueeze(0).unsqueeze(0), 
            size=target_shape,
            mode='bicubic', 
            align_corners=False
        ).squeeze()
        combined += weight * mi_up

    return combined


def compute_radmi_preactivation(
    activations: Dict[str, torch.Tensor],
    target_shape: Tuple[int, int],
    patch_size: int = 4,
    stride: int = 1,
    max_channels: int = 100,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute RADMI using pre-activation features (after Conv2d, before BatchNorm).
    
    This captures negative values and can provide better boundary separation.
    
    Args:
        activations: dict from ActivationCapture with keys like 'up1_conv', 'up2_conv', etc.
        target_shape: (H, W) output size
        
    Returns:
        (H, W) uncertainty map
    """
    if device is None:
        device = activations["up1_conv"].device

    up1 = activations["up1_conv"]
    up2 = activations["up2_conv"]
    up3 = activations["up3_conv"]
    up4 = activations["up4_conv"]

    # same computation as above
    up2_down = F.interpolate(up2, size=(up1.shape[2], up1.shape[3]), 
                              mode='bilinear', align_corners=False)
    mi_12 = compute_layer_mi(up1, up2_down, patch_size, stride, max_channels)

    up3_down = F.interpolate(up3, size=(up2.shape[2], up2.shape[3]), 
                              mode='bilinear', align_corners=False)
    mi_23 = compute_layer_mi(up2, up3_down, patch_size, stride, max_channels)

    up4_down = F.interpolate(up4, size=(up3.shape[2], up3.shape[3]), 
                              mode='bilinear', align_corners=False)
    mi_34 = compute_layer_mi(up3, up4_down, patch_size, stride, max_channels)

    mi_maps = [mi_12, mi_23, mi_34]
    resolutions = [m.shape[0] * m.shape[1] for m in mi_maps]
    total_res = sum(resolutions)
    weights = [r / total_res for r in resolutions]

    combined = torch.zeros(target_shape, device=device)
    for mi_map, weight in zip(mi_maps, weights):
        mi_up = F.interpolate(
            mi_map.unsqueeze(0).unsqueeze(0), 
            size=target_shape,
            mode='bicubic', 
            align_corners=False
        ).squeeze()
        combined += weight * mi_up

    return combined
