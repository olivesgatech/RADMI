# Data

## F3 Netherlands Seismic Dataset

The F3 block is a publicly available 3D seismic volume from the Netherlands North Sea. We use the facies interpretation from Alaudah et al. (2019).

### Download

The preprocessed data can be obtained from the original FaciesNet repository:
https://github.com/olivesgatech/facies_classification_benchmark

After downloading, organize as:
```
data/
├── train/
│   ├── train_seismic.npy   # (401, 701, 255) float32
│   └── train_labels.npy    # (401, 701, 255) int64, classes 0-5
└── test_once/
    ├── test1_seismic.npy   # (200, 701, 255)
    ├── test1_labels.npy
    ├── test2_seismic.npy   # (601, 200, 255)
    └── test2_labels.npy
```

### Classes

| Index | Facies |
|-------|--------|
| 0 | Upper North Sea Group |
| 1 | Middle North Sea Group |
| 2 | Lower North Sea Group |
| 3 | Rijnland/Chalk Group |
| 4 | Scruff Group |
| 5 | Zechstein Group |

### References

Alaudah, Y., Michałowicz, P., Alfarraj, M., & AlRegib, G. (2019). A machine-learning benchmark for facies classification. Interpretation, 7(3), SE175-SE187.
