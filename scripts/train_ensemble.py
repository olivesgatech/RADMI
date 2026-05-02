"""
Train deep ensemble for uncertainty estimation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from RADMI.data import SectionLoader, load_data, standardize_features
from RADMI.models import FaciesSegNet
from RADMI.utils import set_seed, get_device


def train_member(model, train_loader, val_loader, cfg, device, member_idx, save_dir):
    """Train a single ensemble member."""
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = GradScaler()
    criterion_seg = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(cfg["training"]["num_epochs"]):
        # train
        model.train()
        train_loss = 0
        for seismic, labels in train_loader:
            seismic = seismic.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                logits, recon = model(seismic)
                loss_seg = criterion_seg(logits, labels)
                loss_recon = criterion_recon(recon, seismic)
                loss = loss_seg + cfg["training"]["recon_weight"] * loss_recon
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seismic, labels in val_loader:
                seismic = seismic.to(device)
                labels = labels.to(device)
                logits, _ = model(seismic)
                val_loss += criterion_seg(logits, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Member {member_idx} Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, save_dir / f"ensemble_member_{member_idx}_best.pt")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_members", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    n_members = args.n_members or cfg["ensemble"]["n_members"]
    device = get_device()
    print(f"Training {n_members} ensemble members on {device}")
    
    # load data
    train_seismic, train_labels = load_data(
        cfg["data"]["train_seismic"],
        cfg["data"]["train_labels"]
    )
    train_seismic = standardize_features(train_seismic)
    
    save_dir = Path(cfg["paths"]["checkpoint_dir"]) / "deep_ensemble"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for i in range(n_members):
        print(f"\nTraining ensemble member {i+1}/{n_members}")
        
        # different seed per member
        set_seed(42 + i)
        
        # resplit data each time for diversity
        dataset = SectionLoader(train_seismic, train_labels)
        train_size = int(cfg["data"]["train_val_split"] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=2,
        )
        
        model = FaciesSegNet(n_class=cfg["model"]["n_classes"]).to(device)
        
        best_loss = train_member(model, train_loader, val_loader, cfg, device, i, save_dir)
        print(f"  Member {i} best val loss: {best_loss:.4f}")
    
    print(f"\nDone. Saved {n_members} ensemble members to {save_dir}")


if __name__ == "__main__":
    main()
