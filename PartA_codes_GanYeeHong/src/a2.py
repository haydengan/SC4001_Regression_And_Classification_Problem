"""Question A2: ablation experiments over five seeds.

Trains the A1 architecture and the two ablations on combined train+validation
(2017-2021) for the epoch budget selected in A1, then evaluates each once on
the 2022 test year. Reports mean and sample standard deviation across seeds.

    uv run python src/a2.py
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
from matplotlib.ticker import MaxNLocator

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a1 import DATA_PATH, prepare_combined_data, trial_config
from data import prepare_data
from config import Config
from training import evaluate, train_model
from variants import build_variant

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "a2").resolve()
A1_METRICS = (SRC_DIR.parent / "outputs" / "a1" / "metrics.json").resolve()

SEEDS = (42, 43, 44, 45, 46)
VARIANTS = {
    "Baseline (A1)": {"use_categorical": True, "activation": "relu"},
    "(a) No categorical": {"use_categorical": False, "activation": "relu"},
    "(b) Sigmoid": {"use_categorical": True, "activation": "sigmoid"},
}


def run_variant(combined, config, seed, **kwargs):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_variant(combined, config, **kwargs)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = train_model(
        model,
        combined["train"],
        config,
        evaluation_data=combined["test"],
        max_epochs=config.max_epochs,
        early_stopping=False,
        verbose=False,
    )
    rmse, r_squared = evaluate(model, combined["test"], config.batch_size)
    return rmse, r_squared, parameters, history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a1 = json.loads(A1_METRICS.read_text())
    config = trial_config(
        Config(), a1["selected_hidden_width"], a1["selected_embedding_dim"]
    )
    epochs = a1["selected_best_epoch"]
    config = replace(config, max_epochs=epochs)

    combined = prepare_combined_data(DATA_PATH, config)
    print(
        f"width {config.hidden_width}, embedding {config.embedding_dims[0]}, "
        f"{epochs} epochs, seeds {list(SEEDS)}\n"
    )

    records = []
    curves = {}
    for name, kwargs in VARIANTS.items():
        for seed in SEEDS:
            rmse, r_squared, parameters, history = run_variant(
                combined, config, seed, **kwargs
            )
            records.append(
                {
                    "variant": name,
                    "seed": seed,
                    "trainable_parameters": parameters,
                    "test_rmse": rmse,
                    "test_r2": r_squared,
                }
            )
            print(f"{name:<20} seed {seed} | RMSE {rmse:>9,.0f} | R2 {r_squared:.4f}")
            if seed == SEEDS[0]:
                curves[name] = history["evaluation_rmse"]

    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT_DIR / "a2_runs.csv", index=False)

    summary = (
        frame.groupby("variant", sort=False)
        .agg(
            trainable_parameters=("trainable_parameters", "first"),
            rmse_mean=("test_rmse", "mean"),
            rmse_sd=("test_rmse", lambda s: s.std(ddof=1)),
            r2_mean=("test_r2", "mean"),
            r2_sd=("test_r2", lambda s: s.std(ddof=1)),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "a2_summary.csv", index=False)

    print("\n--- A2 summary (mean +/- sample sd over 5 seeds) ---")
    for row in summary.itertuples():
        print(
            f"{row.variant:<20} params {row.trainable_parameters:>6,} | "
            f"RMSE {row.rmse_mean:>9,.0f} +/- {row.rmse_sd:>7,.0f} | "
            f"R2 {row.r2_mean:.4f} +/- {row.r2_sd:.4f}"
        )

    names = summary["variant"].tolist()
    colours = ["#4c78a8", "#f58518", "#54a24b"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].bar(names, summary["rmse_mean"], yerr=summary["rmse_sd"], capsize=5,
                color=colours)
    axes[0].set_ylabel("Test RMSE (SGD)")
    axes[0].set_title("Test RMSE (mean $\\pm$ sd, 5 seeds)")
    axes[1].bar(names, summary["r2_mean"], yerr=summary["r2_sd"], capsize=5,
                color=colours)
    axes[1].set_ylabel("Test $R^2$")
    axes[1].set_title("Test $R^2$ (mean $\\pm$ sd, 5 seeds)")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
        axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_a2_ablation.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    for (name, curve), colour in zip(curves.items(), colours):
        axis.plot(range(1, len(curve) + 1), curve, marker="o", markersize=3,
                  label=name, color=colour)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Test RMSE (SGD)")
    axis.set_title(f"Test RMSE during training (seed {SEEDS[0]})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_a2_curves.png", dpi=200)
    plt.close(figure)

    # ------------------------------------------------------------------
    # Learning curves. The question asks for train and validation metrics
    # against epochs, so these come from a separate run that holds 2021 out,
    # trained exactly as in A1. The two ablations are plotted separately.
    # ------------------------------------------------------------------
    held_out = prepare_data(DATA_PATH, config)
    curves = {}
    for name, kwargs in VARIANTS.items():
        torch.manual_seed(SEEDS[0])
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = build_variant(held_out, config, **kwargs)
        history = train_model(
            model, held_out["train"], replace(config, max_epochs=20),
            evaluation_data=held_out["validation"], early_stopping=True,
            record_r2=True, verbose=False,
        )
        curves[name] = history
        print(f"curve run {name:<20} stopped after {len(history['train_rmse'])} epochs")

    baseline_name = "Baseline (A1)"
    for index, ablation in enumerate(["(a) No categorical", "(b) Sigmoid"], start=1):
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
        for axis, key_tr, key_va, label in (
            (axes[0], "train_fit_rmse", "evaluation_rmse", "RMSE (SGD)"),
            (axes[1], "train_r2", "evaluation_r2", "$R^2$"),
        ):
            for name, colour in ((baseline_name, "#4c78a8"), (ablation, "#e45756")):
                h = curves[name]
                epochs_axis = range(1, len(h[key_tr]) + 1)
                axis.plot(epochs_axis, h[key_tr], color=colour, marker="o",
                          markersize=3, label=f"{name} train")
                axis.plot(epochs_axis, h[key_va], color=colour, marker="o",
                          markersize=3, linestyle=(0, (5, 2)),
                          label=f"{name} validation")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(label)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7, handlelength=3.0)
        figure.suptitle(f"Ablation {ablation} against the baseline", fontsize=10)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"fig_a2_curves_{index}.png", dpi=200)
        plt.close(figure)

    rows = []
    for name, h in curves.items():
        for epoch in range(len(h["train_rmse"])):
            rows.append({
                "variant": name,
                "epoch": epoch + 1,
                "train_rmse": h["train_fit_rmse"][epoch],
                "validation_rmse": h["evaluation_rmse"][epoch],
                "train_r2": h["train_r2"][epoch],
                "validation_r2": h["evaluation_r2"][epoch],
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "a2_curves.csv", index=False)

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "epochs": epochs,
                "hidden_width": config.hidden_width,
                "embedding_dim": config.embedding_dims[0],
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
