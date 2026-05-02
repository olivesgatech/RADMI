"""
Evaluate RADMI and baselines on test set.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from RADMI.data import SectionLoader, load_data, standardize_features
from RADMI.models import FaciesSegNet, FaciesSegNet_MI, FaciesSegNet_Dropout, ActivationCapture
from RADMI.methods import (
    compute_radmi,
    compute_ensemble_uncertainty,
    compute_mc_dropout_uncertainty,
    compute_softmax_entropy,
    compute_msp,
    compute_all_metrics,
    aggregate_metrics,
)
from RADMI.utils import get_device


def load_ensemble(checkpoint_dir: Path, n_classes: int, device: torch.device):
    """Load ensemble models."""
    paths = sorted(checkpoint_dir.glob("ensemble_member_*_best.pt"))
    models = []
    for p in paths:
        model = FaciesSegNet(n_class=n_classes)
        checkpoint = torch.load(p, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        models.append(model)
    print(f"Loaded {len(models)} ensemble members")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--mc_checkpoint", type=str, default=None)
    parser.add_argument("--ensemble_dir", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # load test data
    test_seismic, test_labels = load_data(
        cfg["data"]["test_seismic"],
        cfg["data"]["test_labels"]
    )
    test_seismic = standardize_features(test_seismic)
    
    test_dataset = SectionLoader(test_seismic, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # load models
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    
    # base model for RADMI
    base_path = args.base_checkpoint or checkpoint_dir / "base_model_best.pt"
    model_base = FaciesSegNet_MI(n_class=cfg["model"]["n_classes"])
    model_base.load_state_dict(torch.load(base_path, map_location=device)["model_state_dict"])
    model_base.to(device).eval()
    print(f"Loaded base model from {base_path}")
    
    # MC-Dropout model
    mc_path = args.mc_checkpoint or checkpoint_dir / "mc_dropout_model_best.pt"
    if mc_path.exists():
        checkpoint_mc = torch.load(mc_path, map_location=device)
        model_mc = FaciesSegNet_Dropout(
            n_class=cfg["model"]["n_classes"],
            dropout_rates=checkpoint_mc.get("dropout_rates", cfg["mc_dropout"]["dropout_rates"]),
        )
        model_mc.load_state_dict(checkpoint_mc["model_state_dict"])
        model_mc.to(device)
        print(f"Loaded MC-Dropout model from {mc_path}")
    else:
        model_mc = None
        print("No MC-Dropout model found")
    
    # ensemble
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else checkpoint_dir / "deep_ensemble"
    if ensemble_dir.exists():
        ensemble_models = load_ensemble(ensemble_dir, cfg["model"]["n_classes"], device)
    else:
        ensemble_models = None
        print("No ensemble found")
    
    # run evaluation
    results = {
        "radmi": [],
        "softmax_entropy": [],
        "msp": [],
    }
    if model_mc:
        results["mc_dropout"] = []
    if ensemble_models:
        results["ensemble"] = []
    
    print("\nEvaluating...")
    for seismic, labels in tqdm(test_loader):
        seismic = seismic.to(device)
        
        # RADMI
        with torch.no_grad():
            features = model_base.forward_full(seismic)
            logits, _ = model_base(seismic)
        
        radmi_map = compute_radmi(
            features,
            target_shape=seismic.shape[2:],
            patch_size=cfg["radmi"]["patch_size"],
            stride=cfg["radmi"]["stride"],
            max_channels=cfg["radmi"]["max_channels"],
        ).cpu().numpy()
        
        # baselines
        entropy_map = compute_softmax_entropy(logits)
        msp_map = compute_msp(logits)
        
        results["radmi"].append(radmi_map)
        results["softmax_entropy"].append(entropy_map)
        results["msp"].append(msp_map)
        
        if model_mc:
            mc_map = compute_mc_dropout_uncertainty(
                model_mc, seismic, device, cfg["mc_dropout"]["n_forward"]
            )
            results["mc_dropout"].append(mc_map)
        
        if ensemble_models:
            ens_map = compute_ensemble_uncertainty(ensemble_models, seismic, device)
            results["ensemble"].append(ens_map)
    
    # compute metrics if we have ensemble as reference
    if ensemble_models:
        print("\nComputing metrics vs ensemble...")
        
        method_metrics = {}
        for method in ["radmi", "softmax_entropy", "msp", "mc_dropout"]:
            if method not in results:
                continue
            
            metrics_list = []
            for i in range(len(results[method])):
                m = compute_all_metrics(results[method][i], results["ensemble"][i])
                metrics_list.append(m)
            
            method_metrics[method] = aggregate_metrics(metrics_list)
        
        # print results
        print("\n" + "=" * 80)
        print("RESULTS vs DEEP ENSEMBLE")
        print("=" * 80)
        
        headers = ["Method", "Pearson↑", "Spearman↑", "mIoU↑", "DICE↑", "JS Div↓"]
        print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12}")
        print("-" * 80)
        
        for method, metrics in method_metrics.items():
            row = f"{method:<20}"
            for key in ["pearson", "spearman", "miou", "dice", "js_div"]:
                mean, std = metrics[key]
                row += f" {mean:>6.4f}±{std:<4.4f}"
            print(row)
    
    # save results
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)
    
    for method, maps in results.items():
        np.save(results_dir / f"{method}_maps.npy", np.stack(maps))
    
    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()
