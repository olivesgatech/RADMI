"""
Visualization utilities for comparing uncertainty methods.

Implements a cleaned version of your `compare_all_uncertainty_methods` routine:
- shows GT, prediction, ensemble, RADMI, entropy, 1-MSP, MC-dropout, prediction switches
- saves a grid figure to disk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


@dataclass(frozen=True)
class FaciesVizConfig:
    class_colors: Sequence[str] = ("#ffb400", "#fee401", "#f9ff00", "#00ff6d", "#7dffff", "#ffd39b")
    class_names: Sequence[str] = ("Upper NS", "Middle NS", "Lower NS", "Rijnland/Chalk", "Scruff", "Zechstein")
    cmap_uncertainty: str = "turbo"


def plot_method_grid(
    *,
    label: np.ndarray,
    pred: np.ndarray,
    ensemble: np.ndarray,
    radmi: np.ndarray,
    entropy: np.ndarray,
    one_minus_msp: np.ndarray,
    mc_dropout: np.ndarray,
    pred_switches: Optional[np.ndarray] = None,
    title_left: str = "",
    save_path: Optional[str] = None,
    viz_cfg: FaciesVizConfig = FaciesVizConfig(),
    rotate_k: int = -1,
) -> plt.Figure:
    """
    Create a single-row grid like your paper figure examples.

    All arrays expected to be (H,W). If `rotate_k` is -1, we rotate as in your notebook.
    """
    maps: Dict[str, np.ndarray] = {
        "Ground Truth": label,
        "Prediction": pred,
        "Deep Ensemble": ensemble,
        "RADMI (Ours)": radmi,
        "Softmax Entropy": entropy,
        "1 - MSP": one_minus_msp,
        "MC-Dropout": mc_dropout,
    }
    if pred_switches is not None:
        maps["Pred. Switches"] = pred_switches

    # setup seg colormap
    cmap_seg = ListedColormap(list(viz_cfg.class_colors))
    norm_seg = BoundaryNorm(boundaries=np.arange(len(viz_cfg.class_colors) + 1) - 0.5, ncolors=len(viz_cfg.class_colors))

    n_cols = len(maps)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.2))
    if n_cols == 1:
        axes = [axes]

    # rotate helper
    def rot(x: np.ndarray) -> np.ndarray:
        return np.rot90(x, k=rotate_k) if rotate_k is not None else x

    im_handles = {}

    for ax, (name, arr) in zip(axes, maps.items()):
        if name in ("Ground Truth", "Prediction"):
            im = ax.imshow(rot(arr), cmap=cmap_seg, norm=norm_seg, aspect="auto")
        else:
            im = ax.imshow(rot(arr), cmap=viz_cfg.cmap_uncertainty, aspect="auto")
        im_handles[name] = im
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    if title_left:
        fig.suptitle(title_left, fontsize=12)

    # Add 2 colorbars: one for facies, one for uncertainty
    # Facies bar
    cbar_ax1 = fig.add_axes([0.92, 0.60, 0.012, 0.30])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_seg, cmap=cmap_seg), cax=cbar_ax1, ticks=np.arange(len(viz_cfg.class_names)))
    cb1.ax.set_yticklabels(list(viz_cfg.class_names), fontsize=8)
    cb1.ax.invert_yaxis()
    cb1.ax.set_title("Facies", fontsize=9, pad=4)

    # Uncertainty bar (use RADMI handle)
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.012, 0.30])
    cb2 = fig.colorbar(im_handles["RADMI (Ours)"], cax=cbar_ax2)
    cb2.ax.set_title("Unc.", fontsize=9, pad=4)
    cb2.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig