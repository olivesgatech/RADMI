"""
Train model with dropout for MC-Dropout inference.
"""

import argparse
from pathlib import Path

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
from RADMI.models import FaciesSegNet_Dropout
from RADMI.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # load data
    train_seismic, train_labels = load_data(
        cfg["data"]["train_seismic"],
        cfg["data"]["train_labels"]
    )
    train_seismic = standardize_features(train_seismic)
    
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
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # model with dropout
    model = FaciesSegNet_Dropout(
        n_class=cfg["model"]["n_classes"],
        dropout_rates=cfg["mc_dropout"]["dropout_rates"],
    )
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = GradScaler()
    criterion_seg = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    
    save_dir = Path(cfg["paths"]["checkpoint_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(cfg["training"]["num_epochs"]):
        # train
        model.train()
        train_loss = 0
        for seismic, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
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
        
        print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dropout_rates": cfg["mc_dropout"]["dropout_rates"],
            }, save_dir / "mc_dropout_model_best.pt")
    
    torch.save({
        "epoch": cfg["training"]["num_epochs"],
        "model_state_dict": model.state_dict(),
        "dropout_rates": cfg["mc_dropout"]["dropout_rates"],
    }, save_dir / f"mc_dropout_model_final_epoch{cfg['training']['num_epochs']}.pt")
    
    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
