"""Question A4: Captum FeatureAblation attributions for the A1 and A3 models.

Both models are trained with the same seed on combined train+validation, then
each of the ten original input features is ablated on every test sample. The
global importance of a feature is the mean absolute attribution across the test
set, in Singapore dollars.

Categorical features are ablated to index 0, the reserved "unseen category"
row; continuous features are ablated to 0, which is the training mean after
standardisation.

    uv run python src/a4.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from captum.attr import FeatureAblation
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a1 import DATA_PATH, prepare_combined_data, trial_config
from config import Config
from training import train_model
from variants import build_variant

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "a4").resolve()
A1_METRICS = (SRC_DIR.parent / "outputs" / "a1" / "metrics.json").resolve()
A3_METRICS = (SRC_DIR.parent / "outputs" / "a3" / "metrics.json").resolve()

SEED = 42
ATTRIBUTION_BATCH = 2048
# The question asks for the two models only. When lambda is 0 they are the same
# architecture, and the TA confirmed one attribution is then enough, so the
# supplementary non-zero-lambda panel is off.
INCLUDE_SUPPLEMENTARY = False


def feature_names(config: Config):
    return list(config.categorical_features) + list(config.continuous_features)


def attribute(model, dataset, config):
    """Mean absolute FeatureAblation attribution per original feature, in SGD."""
    model.eval()
    categorical_count = len(config.categorical_features)
    continuous_count = len(config.continuous_features)

    def forward(categorical, continuous):
        return model.predict_price(categorical, continuous)

    ablator = FeatureAblation(forward)
    categorical_mask = torch.arange(categorical_count).unsqueeze(0)
    continuous_mask = (
        torch.arange(continuous_count) + categorical_count
    ).unsqueeze(0)

    totals = torch.zeros(categorical_count + continuous_count)
    rows = 0
    for categorical, continuous, _ in DataLoader(dataset, batch_size=ATTRIBUTION_BATCH):
        categorical_attribution, continuous_attribution = ablator.attribute(
            inputs=(categorical, continuous),
            baselines=(0, 0.0),
            feature_mask=(categorical_mask, continuous_mask),
        )
        batch = torch.cat(
            [categorical_attribution.float(), continuous_attribution], dim=1
        )
        totals += batch.abs().sum(dim=0)
        rows += len(categorical)
    return (totals / rows).tolist()


def train_for_attribution(combined, config, epochs, wide_lambda=None):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_variant(combined, config, wide_lambda=wide_lambda)
    train_model(
        model, combined["train"], config, evaluation_data=None,
        max_epochs=epochs, early_stopping=False, verbose=False,
    )
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a1 = json.loads(A1_METRICS.read_text())
    a3 = json.loads(A3_METRICS.read_text())
    epochs = a1["selected_best_epoch"]
    config = replace(
        trial_config(Config(), a1["selected_hidden_width"], a1["selected_embedding_dim"]),
        max_epochs=epochs,
    )
    selected_lambda = a3["selected_lambda"]

    combined = prepare_combined_data(DATA_PATH, config)
    names = feature_names(config)

    # The lambda selected in A3 is 0, which switches the linear branch off, so
    # the two required models are architecturally identical. The best non-zero
    # lambda is attributed as well, to give a comparison in which the wide
    # branch is actually active.
    non_zero = min(
        (row for row in a3["lambda_search"] if row["lambda"] > 0),
        key=lambda row: row["best_validation_rmse"],
    )["lambda"]

    models = {
        "Baseline (A1)": train_for_attribution(combined, config, epochs),
        f"Wide-and-deep (A3)\n$\\lambda$ = {selected_lambda:g}": train_for_attribution(
            combined, config, epochs, wide_lambda=selected_lambda
        ),
    }
    if INCLUDE_SUPPLEMENTARY and selected_lambda == 0:
        models[f"Wide-and-deep (suppl.)\n$\\lambda$ = {non_zero:g}"] = (
            train_for_attribution(combined, config, epochs, wide_lambda=non_zero)
        )

    scores = {}
    for label, model in models.items():
        scores[label] = attribute(model, combined["test"], config)
        ranked = sorted(zip(names, scores[label]), key=lambda pair: -pair[1])
        print(f"\n{label} — top 3: " + ", ".join(
            f"{name} (SGD {value:,.0f})" for name, value in ranked[:3]
        ))

    frame = pd.DataFrame({"feature": names, **scores})
    frame.to_csv(OUTPUT_DIR / "a4_feature_importance.csv", index=False)
    print()
    print(frame.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    figure, axes = plt.subplots(1, len(scores), figsize=(2.85 * len(scores), 3.5))
    for axis, (label, values) in zip(axes, scores.items()):
        order = sorted(range(len(names)), key=lambda i: values[i])
        colour = "#4c78a8" if "Baseline" in label else (
            "#e45756" if "suppl." not in label else "#72b7b2"
        )
        axis.barh([names[i] for i in order], [values[i] for i in order], color=colour)
        axis.set_xlabel("Mean |attribution| (SGD)", fontsize=9)
        axis.set_title(label, fontsize=8.5)
        axis.tick_params(axis="y", labelsize=7.5)
        axis.tick_params(axis="x", labelsize=7.5)
        axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_a4_feature_importance.png", dpi=200)
    plt.close(figure)

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "selected_lambda": selected_lambda,
                "supplementary_lambda": non_zero,
                "features": names,
                "scores": scores,
                "top3": {
                    label: [
                        name
                        for name, _ in sorted(
                            zip(names, values), key=lambda pair: -pair[1]
                        )[:3]
                    ]
                    for label, values in scores.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
