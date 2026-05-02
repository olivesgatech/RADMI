"""
FaciesSegNet variants for feature extraction and uncertainty estimation.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .faciessegnet import FaciesSegNet


class FaciesSegNet_MI(FaciesSegNet):
    """FaciesSegNet with forward_full for extracting intermediate features."""

    def forward_full(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return features at each encoder/decoder block."""
        features = {}
        skips = []
        
        for i, block in enumerate(self.down_convs):
            x = block(x)
            features[f"down{i+1}"] = x
            skips.append(x)
            if i < len(self.down_convs) - 1:
                x = self.maxpool(x)
        
        for i, block in enumerate(self.up_convs):
            x = block(x)
            features[f"up{i+1}"] = x
        
        return features


class FaciesSegNet_Dropout(FaciesSegNet):
    """FaciesSegNet with dropout in decoder for MC-Dropout inference."""

    def __init__(
        self, 
        n_class: int, 
        out_channels: tuple = (8, 10, 30, 40, 60),
        dropout_rates: List[float] = None,
    ):
        super().__init__(n_class, out_channels)
        if dropout_rates is None:
            dropout_rates = [0.1, 0.1, 0.1, 0.1]
        self.dropout_rates = dropout_rates

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2], x.shape[3]

        for i, block in enumerate(self.down_convs):
            x = block(x)
            if i < len(self.down_convs) - 1:
                x = self.maxpool(x)

        for i, block in enumerate(self.up_convs):
            x = block(x)
            if i < len(self.dropout_rates):
                x = F.dropout2d(x, p=self.dropout_rates[i], training=self.training)

        out = self.conv_last(x)[:, :, :h, :w]
        reconstruct = self.conv_reconstruct(x)[:, :, :h, :w]

        return out, reconstruct

    def forward_full(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        
        for i, block in enumerate(self.down_convs):
            x = block(x)
            features[f"down{i+1}"] = x
            if i < len(self.down_convs) - 1:
                x = self.maxpool(x)
        
        for i, block in enumerate(self.up_convs):
            x = block(x)
            if i < len(self.dropout_rates):
                x = F.dropout2d(x, p=self.dropout_rates[i], training=self.training)
            features[f"up{i+1}"] = x
        
        return features


class ActivationCapture:
    """
    Capture activations at specific layers using forward hooks.
    
    Useful for getting pre-activation features (after Conv2d, before BatchNorm).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.hooks = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def register_decoder_hooks(self):
        """
        Register hooks for decoder blocks.
        
        Captures both post-conv (pre-batchnorm) and post-relu activations.
        Block structure: ConvTranspose -> ReLU -> Conv2d -> BatchNorm -> ReLU
        """
        self.clear()
        
        for i, up_block in enumerate(self.model.up_convs):
            name = f"up{i+1}"
            # after Conv2d (index 2), before BatchNorm
            hook = up_block[2].register_forward_hook(self._make_hook(f"{name}_conv"))
            self.hooks.append(hook)
            # after final ReLU (index 4)
            hook = up_block[4].register_forward_hook(self._make_hook(f"{name}_relu"))
            self.hooks.append(hook)

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def forward(self, x: torch.Tensor):
        """Run forward pass and return output + captured activations."""
        self.activations = {}
        with torch.no_grad():
            output = self.model(x)
        return output, self.activations
