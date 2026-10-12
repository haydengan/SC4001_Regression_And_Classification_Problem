# SC4001 Part B — chest X-ray pneumonia classification

Code for Questions B1–B4 of Part B.

## Environment

- Python 3.12 (pinned in `.python-version`)
- CPU only, no GPU required; the complete pipeline runs in roughly 40 minutes on four cores
- Dependencies managed with [uv](https://docs.astral.sh/uv/), exact versions locked in `uv.lock`

| Package | Version |
| --- | --- |
| torch | 2.14.1 |
| captum | 0.9.0 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |

```bash
uv sync
```

## Data

`chestxray_binary_64.npz`, 64×64 greyscale chest X-rays, already partitioned. The scripts
look for it in the project root and then one directory above, so either location works.

| Partition | Images | Healthy (0) | Pneumonia (1) |
| --- | --- | --- | --- |
| Train | 4,708 | 1,214 | 3,494 |
| Validation | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |

Intensities are divided by 255 to reach [0, 1] and then standardised with the mean
(0.5717) and standard deviation (0.1741) of the **training** partition only.

Class 1 is pneumonia, so sensitivity is recall on pneumonia and specificity is recall on
the healthy class.

## Seeds

Seed **42** throughout, applied with `torch.manual_seed(42)` before every model
construction, so all runs in a sweep start from identical initial weights.
`torch.use_deterministic_algorithms(True, warn_only=True)` is enabled. The five Grad-CAM
images in B4 are drawn with a `torch.Generator` seeded with 42.

## Commands

Run from the project root, in this order — each question reads the selections written by
the previous one.

```bash
uv run python src/b1.py   # AlexNet-like, learning-rate search
uv run python src/b2.py   # VGG-like, batch-size search
uv run python src/b3.py   # ResNet-like and the B1-B3 comparison
uv run python src/b4.py   # Grad-CAM on all three architectures
```

## Files

| Path | Role |
| --- | --- |
| `src/data.py` | loads the npz, scales to [0, 1], standardises with training statistics |
| `src/models.py` | the three Figure B architectures, parameter and FLOP counters |
| `src/training.py` | training loop, early stopping, and the classification metrics |
| `src/b1.py` … `src/b4.py` | one script per question |

Each script writes to `outputs/b1` … `outputs/b4`: a `metrics.json` holding every number
quoted in the report, the figures, and the trained weights (`b1_model.pt`, `b2_model.pt`,
`b3_model.pt`). B4 loads those weights, so the explanations describe exactly the models
reported in B1–B3.

## Architectures

Transcribed from Figure B. All convolutions use SAME padding, BatchNorm and no bias;
fully connected layers keep their bias; layers marked /2 use stride 2.

- **B1 (AlexNet-like)** — 5×5 conv 16, pool/2, 5×5 conv 32, pool/2, 3×3 conv 64, pool/2,
  average pool to 4×4, fc 64, fc 32, fc 1. Dropout 0.5 after each hidden fc layer only.
- **B2 (VGG-like)** — (3×3 conv 8) ×2, pool/2, (3×3 conv 16) ×2, pool/2, (3×3 conv 32) ×2,
  pool/2, average pool to 1×1, fc 1.
- **B3 (ResNet-like)** — 3×3 conv 16, pool/2, then three residual blocks of two 3×3
  convolutions each at widths 16, 32 and 64. The second and third blocks downsample with
  stride 2 and carry a 1×1 stride-2 convolution on the skip path. Average pool to 1×1, fc 1.

One interpretation note: in B3 the second convolution of each block is followed by
BatchNorm only, with the ReLU applied after the residual addition. This is the standard
residual design and is what the figure caption means by "ReLU … including after residual
addition".

## Protocol

Adam, `BCEWithLogitsLoss` on a single output logit, at most 20 epochs, early stopping with
patience 4 and minimum improvement 1e-4, as the assignment brief specifies. Early stopping
monitors validation **loss** and restores the best weights; model selection within each
sweep then uses the lowest validation **classification error**, as each question requires.

Once a sweep has chosen its hyperparameter, the reported model is a **fresh network
retrained from scratch on the combined training and validation data** (5,232 images) for
the epoch count found during the search, with early stopping switched off. That retrained
model is then evaluated on the test set exactly once. The standardisation statistics are
not refitted on the combined set: the brief specifies normalising with the mean and
standard deviation of the training set, so those are kept throughout.

The training set is 74% pneumonia and no class weighting is applied.

## Results

| | B1 AlexNet-like | B2 VGG-like | B3 ResNet-like |
| --- | --- | --- | --- |
| Setting | lr 1e-3, batch 128 | lr 1e-3, batch 128 | lr 1e-3, batch 128 |
| Retrain epochs | 15 | 14 | 10 |
| Test accuracy | 0.8269 | 0.7949 | **0.9311** |
| Test sensitivity | 0.9974 | 1.0000 | 0.9590 |
| Test specificity | 0.5427 | 0.4530 | **0.8846** |
| Classification error | 0.1731 | 0.2051 | **0.0689** |
| Learnable parameters | 99,569 | 18,185 | 77,169 |
| FLOPs per image | 39.06 M | 19.46 M | 25.30 M |

Selected hyperparameters: learning rate **1e-3** (B1), batch size **128** (B2).

FLOPs count the forward pass through convolution and linear layers, with one
multiply-accumulate counted as two operations.

## Reproducibility

Verified by deleting `outputs/` and re-running all four scripts: the resulting
`metrics.json` files are identical to those committed here.
