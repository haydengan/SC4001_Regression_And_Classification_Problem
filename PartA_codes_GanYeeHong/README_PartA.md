# SC4001 Part A — HDB resale price regression

Code for Questions A1–A4 of Part A.

## Environment

- Python 3.12 (pinned in `.python-version`)
- CPU only, no GPU required; the complete pipeline runs in about three minutes on four cores
- Dependencies managed with [uv](https://docs.astral.sh/uv/), exact versions locked in `uv.lock`

| Package | Version |
| --- | --- |
| torch | 2.13.0 |
| ray[tune] | 2.58.0 |
| captum | 0.9.0 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |

```bash
uv sync
```

## Data

`hdb_price_prediction.csv` sits in the repository root (159,553 rows).

| Partition | Years | Rows |
| --- | --- | --- |
| Train | 2017–2020 | 87,370 |
| Validation | 2021 | 29,057 |
| Test | 2022 | 26,702 |

2023 rows are present in the CSV and are not used.

## Seeds

- A1 and the A3 λ search use seed **42** (`Config.seed` in `src/config.py`).
- A2 and the five-seed evaluations in A3 use seeds **42, 43, 44, 45, 46**.
- A4 trains both models with seed **42**, as the question requires.
- `torch.manual_seed` is called before every model construction, including inside each
  Ray Tune trial, and `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled.
- Each Ray trial runs single-threaded (`torch.set_num_threads(1)`) so results do not
  depend on how many trials run in parallel.

## Commands

Run from the repository root, in this order — each question reads the selections written
by the previous one.

```bash
uv run python src/a1.py                          # grid search, retrain, test metrics
uv run python src/a1_diagram.py                  # Figure 1
uv run python src/a2.py                          # ablations over five seeds
uv run python src/a3.py                          # lambda search and wide-and-deep
uv run python src/a4.py                          # Captum FeatureAblation

uv run python src/baseline.py                    # provided baseline, unchanged
uv run python src/a1_probe_test_early_stopping.py  # diagnostic, see below
```

## Files

| Path | Role |
| --- | --- |
| `src/config.py`, `src/data.py`, `src/model.py`, `src/baseline.py` | provided, unmodified |
| `src/training.py` | training loop with per-epoch history, optional early stopping, RMSE / R² evaluation, residual decomposition |
| `src/variants.py` | model variants: A2 ablations and the A3 wide-and-deep architecture |
| `src/a1.py` | A1 pipeline |
| `src/a1_diagram.py` | A1 block diagram |
| `src/a2.py` | A2 ablations |
| `src/a3.py` | A3 wide-and-deep |
| `src/a4.py` | A4 attributions |
| `src/a1_probe_test_early_stopping.py` | diagnostic: measures how much early stopping monitored on the test year would flatter the A1 result; not used for any reported number |

The four provided files are unmodified, verified byte-for-byte against
`housing_baseline_data.zip`.

## Outputs

Written to `outputs/a1` … `outputs/a4`. Each directory holds a `metrics.json` with the
numbers quoted in the report, the per-run CSVs behind them, and the report figures:

| Figure | File |
| --- | --- |
| 1 | `outputs/a1/fig_a1_block_diagram.png` |
| 2 | `outputs/a1/fig_a1_curves.png` |
| 3 | `outputs/a2/fig_a2_ablation.png` |
| 4 | `outputs/a3/fig_a3_wide_and_deep.png` |
| 5 | `outputs/a4/fig_a4_feature_importance.png` |

## Protocol

Adam, learning rate 1e-2, batch size 512, at most 20 epochs, early stopping with patience
4 and minimum improvement 1e-4, as specified in the assignment brief.

Early stopping is used wherever a held-out split exists: the A1 grid search and the A3 λ
search, both of which train on 2017–2020 and monitor 2021. For every retrain on combined
train+validation there is no held-out split left to monitor, so the epoch budget is fixed
to the 13 epochs at which the selected A1 configuration reached its best validation RMSE.
That same budget is used for A2, A3 and A4 so the comparisons are like-for-like. Category
encoders and standardisation statistics are refitted on whichever rows the model is
trained on. The 2022 test year is used only for final evaluation.

## Results

**A1** — baseline 3,505 trainable parameters. Selected hidden width 128, embedding
dimension 12 (8,649 parameters), best validation RMSE SGD 60,111 at epoch 13.
Test RMSE SGD 92,163, R² 0.7070.

**A2** — mean ± sample standard deviation over five seeds:

| Model | Params | Test RMSE (SGD) | Test R² |
| --- | --- | --- | --- |
| Baseline (A1) | 8,649 | 95,638 ± 3,316 | 0.6842 ± 0.0220 |
| (a) No categorical | 1,281 | 111,010 ± 5,738 | 0.5740 ± 0.0437 |
| (b) Sigmoid | 8,649 | 94,913 ± 7,886 | 0.6875 ± 0.0504 |

**A3** — λ selected on 2021 validation: λ = 0, i.e. the linear branch is switched off.
Wide-and-deep test RMSE SGD 96,779 ± 3,532, R² 0.6766 ± 0.0231, which is
+1,141 SGD against a standard error of 2,167 SGD and so indistinguishable from the
baseline.

**A4** — the three most important features are `floor_area_sqm`, `flat_model_type` and
`town` for both required models. The two MRT graph features are the weakest
(`degree_centrality` SGD 3,325, `eigenvector_centrality` SGD 5,959).

## Reproducibility

Verified by deleting `outputs/` and running the pipeline from scratch: the resulting
`metrics.json` files are identical to those committed here.
