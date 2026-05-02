"""
Metrics for comparing uncertainty maps.
"""

from typing import Dict, List
import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from scipy.spatial.distance import cosine, jensenshannon
from scipy.special import kl_div
from scipy.ndimage import distance_transform_edt


def compute_miou(a: np.ndarray, b: np.ndarray, n_bins: int = 100) -> float:
    """Mean IoU across thresholds."""
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-10)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-10)

    thresholds = np.linspace(0, 1, n_bins)
    ious = []

    for t in thresholds:
        a_bin = (a_norm >= t).astype(float)
        b_bin = (b_norm >= t).astype(float)
        
        intersection = np.sum(a_bin * b_bin)
        union = np.sum(a_bin) + np.sum(b_bin) - intersection
        
        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def compute_dice(a: np.ndarray, b: np.ndarray, n_bins: int = 100) -> float:
    """Mean DICE coefficient across thresholds."""
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-10)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-10)

    thresholds = np.linspace(0, 1, n_bins)
    dices = []

    for t in thresholds:
        a_bin = (a_norm >= t).astype(float)
        b_bin = (b_norm >= t).astype(float)
        
        intersection = np.sum(a_bin * b_bin)
        total = np.sum(a_bin) + np.sum(b_bin)
        
        if total > 0:
            dices.append(2 * intersection / total)

    return np.mean(dices) if dices else 0.0


def compute_chamfer(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    """Bidirectional Chamfer distance."""
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-10)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-10)

    a_bin = (a_norm >= threshold).astype(bool)
    b_bin = (b_norm >= threshold).astype(bool)

    if not np.any(a_bin) or not np.any(b_bin):
        return np.nan

    dist_a = distance_transform_edt(~a_bin)
    dist_b = distance_transform_edt(~b_bin)

    chamfer_a_to_b = np.mean(dist_b[a_bin])
    chamfer_b_to_a = np.mean(dist_a[b_bin])

    return (chamfer_a_to_b + chamfer_b_to_a) / 2


def compute_emd(a: np.ndarray, b: np.ndarray, n_bins: int = 100) -> float:
    """Earth mover's distance between histograms."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    bins = np.linspace(0, 1, n_bins + 1)
    a_norm = (a_flat - a_flat.min()) / (a_flat.max() - a_flat.min() + 1e-10)
    b_norm = (b_flat - b_flat.min()) / (b_flat.max() - b_flat.min() + 1e-10)

    hist_a, _ = np.histogram(a_norm, bins=bins, density=True)
    hist_b, _ = np.histogram(b_norm, bins=bins, density=True)

    hist_a = hist_a / (hist_a.sum() + 1e-10)
    hist_b = hist_b / (hist_b.sum() + 1e-10)

    return wasserstein_distance(hist_a, hist_b)


def compute_all_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compute all comparison metrics between two heatmaps."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    if a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
        return {k: np.nan for k in [
            'pearson', 'spearman', 'cosine_sim',
            'kl_div', 'js_div', 'l2_dist',
            'miou', 'dice', 'chamfer', 'emd'
        ]}

    metrics = {}

    # correlation (higher = more similar)
    metrics['pearson'], _ = pearsonr(a_flat, b_flat)
    metrics['spearman'], _ = spearmanr(a_flat, b_flat)
    metrics['cosine_sim'] = 1 - cosine(a_flat, b_flat)

    # distance (lower = more similar)
    a_pos = a_flat - a_flat.min() + 1e-10
    b_pos = b_flat - b_flat.min() + 1e-10
    a_prob = a_pos / a_pos.sum()
    b_prob = b_pos / b_pos.sum()

    metrics['kl_div'] = np.sum(kl_div(a_prob, b_prob))
    metrics['js_div'] = jensenshannon(a_prob, b_prob)

    a_normed = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-10)
    b_normed = (b_flat - b_flat.mean()) / (b_flat.std() + 1e-10)
    metrics['l2_dist'] = np.linalg.norm(a_normed - b_normed) / np.sqrt(len(a_flat))

    # overlap
    metrics['miou'] = compute_miou(a, b)
    metrics['dice'] = compute_dice(a, b)
    metrics['chamfer'] = compute_chamfer(a, b)
    metrics['emd'] = compute_emd(a, b)

    return metrics


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, tuple]:
    """Aggregate list of metric dicts into mean ± std."""
    keys = metric_list[0].keys()
    agg = {}
    
    for k in keys:
        vals = [m[k] for m in metric_list if np.isfinite(m[k])]
        if vals:
            agg[k] = (np.mean(vals), np.std(vals))
        else:
            agg[k] = (np.nan, np.nan)
    
    return agg
