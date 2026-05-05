# RADMI: Resolution-Aggregated Decoder Mutual Information

[![arXiv](https://img.shields.io/badge/arXiv-2605.01502-b31b1b.svg)](https://arxiv.org/abs/2605.01502)

Uncertainty estimation for semantic segmentation using mutual information between decoder layers.

## Overview

RADMI computes uncertainty maps by measuring statistical dependence between consecutive decoder activations. At class boundaries, conflicting context forces the decoder to produce higher-variance activations with stronger inter-layer dependence. MI captures this phenomenon as an uncertainty signal.

Key features:
- Single forward pass (no sampling or ensembles at inference)
- Works with any pretrained encoder-decoder
- No architectural modifications required

## Installation

```bash
git clone https://github.com/olivesgatech/RADMI.git
cd RADMI
pip install -r requirements.txt
```

## Data

Download the F3 seismic dataset and place it in the `data/` directory:
```
data/
├── train/
│   ├── train_seismic.npy
│   └── train_labels.npy
└── test_once/
    ├── test1_seismic.npy
    └── test1_labels.npy
```

See `data/README.md` for download instructions.

## Usage

### Training

Train base model:
```bash
python scripts/train_base.py --config configs/default.yaml
```

Train deep ensemble (for comparison):
```bash
python scripts/train_ensemble.py --config configs/default.yaml --n_members 30
```

Train MC-Dropout model:
```bash
python scripts/train_mc_dropout.py --config configs/default.yaml
```

### Evaluation

Run RADMI and baselines on test set:
```bash
python scripts/evaluate.py --config configs/default.yaml
```

### Using RADMI in your code

```python
from RADMI.models import FaciesSegNet_MI
from RADMI.methods import compute_radmi

model = FaciesSegNet_MI(n_class=6)
model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])
model.eval()

with torch.no_grad():
    features = model.forward_full(input_tensor)
    
uncertainty_map = compute_radmi(
    features,
    target_shape=input_tensor.shape[2:],
    patch_size=4,
    stride=1,
)
```

## Project Structure

```
RADMI/
├── src/RADMI/
│   ├── data/          # Data loading and augmentation
│   ├── methods/       # RADMI, baselines, metrics
│   ├── models/        # FaciesSegNet architecture
│   └── utils/         # Helpers
├── scripts/           # Training and evaluation
├── configs/           # Configuration files
├── checkpoints/       # Saved models
└── results/           # Output
```

## Citation

```bibtex
@inproceedings{stevens2026radmi,
  title={RADMI: Latent Information Aggregation as a Proxy for Model Uncertainty},
  author={Stevens, William and Prabhushankar, Mohit and AlRegib, Ghassan},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2026}
}
```

## Links

Paper: https://arxiv.org/abs/2605.01502

Associated Website: https://alregib.ece.gatech.edu/

## License

MIT

## Acknowledgments

This work is supported by the ML4Seismic Consortium at Georgia Tech.
