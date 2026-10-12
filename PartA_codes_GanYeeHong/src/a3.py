"""Question A3: wide-and-deep architecture.

Selects lambda on the 2021 validation year (training on 2017-2020), then
evaluates the selected wide-and-deep model over five seeds on combined
train+validation, using the A1 hyperparameters and epoch budget throughout so
the comparison against A2's baseline is like-for-like.

    uv run python src/a3.py
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

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a1 import DATA_PATH, prepare_combined_data, trial_config
from config import Config
from data import prepare_data
from training import evaluate, train_model
from variants import build_variant

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "a3").resolve()
A1_METRICS = (SRC_DIR.parent / "outputs" / "a1" / "metrics.json").resolve()
A2_METRICS = (SRC_DIR.parent / "outputs" / "a2" / "metrics.json").resolve()

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
SEEDS = (42, 43, 44, 45, 46)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a1 = json.loads(A1_METRICS.read_text())
    base = trial_config(
        Config(), a1["selected_hidden_width"], a1["selected_embedding_dim"]
    )
    # Epoch budget comes from the lambda search itself: the best epoch of the
    # selected lambda, as different lambdas peak at different epochs.

    # Stage 1: select lambda on the held-out 2021 validation year.
    data = prepare_data(DATA_PATH, base)
    search = []
    search_curves = {}
    for value in LAMBDAS:
        torch.manual_seed(base.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = build_variant(data, base, wide_lambda=value)
        history = train_model(
            model, data["train"], base, evaluation_data=data["validation"],
            early_stopping=True, verbose=False,
        )
        search.append(
            {
                "lambda": value,
                "best_epoch": history["best_epoch"],
                "best_validation_rmse": history["best_evaluation_rmse"],
            }
        )
        search_curves[value] = history["evaluation_rmse"]
        print(
            f"lambda {value:<5} | best validation RMSE "
            f"{history['best_evaluation_rmse']:>9,.0f} at epoch {history['best_epoch']}"
        )

    search_frame = pd.DataFrame(search)
    search_frame.to_csv(OUTPUT_DIR / "a3_lambda_search.csv", index=False)
    best = search_frame.loc[search_frame["best_validation_rmse"].idxmin()]
    selected_lambda = float(best["lambda"])
    epochs = int(best["best_epoch"])
    print(f"\nselected lambda = {selected_lambda} (best epoch {epochs})")

    # Stage 2: five seeds on combined train+validation, A1 epoch budget.
    config = replace(base, max_epochs=epochs)
    combined = prepare_combined_data(DATA_PATH, config)
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = build_variant(combined, config, wide_lambda=selected_lambda)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_model(
            model, combined["train"], config, evaluation_data=None,
            max_epochs=epochs, early_stopping=False, verbose=False,
        )
        rmse, r_squared = evaluate(model, combined["test"], config.batch_size)
        records.append(
            {
                "seed": seed,
                "trainable_parameters": parameters,
                "test_rmse": rmse,
                "test_r2": r_squared,
            }
        )
        print(f"wide-and-deep seed {seed} | RMSE {rmse:>9,.0f} | R2 {r_squared:.4f}")

    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT_DIR / "a3_runs.csv", index=False)

    wide = {
        "name": "Wide-and-deep (A3)",
        "trainable_parameters": int(frame["trainable_parameters"].iloc[0]),
        "rmse_mean": frame["test_rmse"].mean(),
        "rmse_sd": frame["test_rmse"].std(ddof=1),
        "r2_mean": frame["test_r2"].mean(),
        "r2_sd": frame["test_r2"].std(ddof=1),
    }
    a2 = json.loads(A2_METRICS.read_text())
    baseline = next(r for r in a2["summary"] if r["variant"] == "Baseline (A1)")

    print("\n--- A3 vs A1 baseline (mean +/- sample sd over 5 seeds) ---")
    for label, rmse_mean, rmse_sd, r2_mean, r2_sd in (
        ("Baseline (A1)", baseline["rmse_mean"], baseline["rmse_sd"],
         baseline["r2_mean"], baseline["r2_sd"]),
        (wide["name"], wide["rmse_mean"], wide["rmse_sd"],
         wide["r2_mean"], wide["r2_sd"]),
    ):
        print(
            f"{label:<20} RMSE {rmse_mean:>9,.0f} +/- {rmse_sd:>7,.0f} | "
            f"R2 {r2_mean:.4f} +/- {r2_sd:.4f}"
        )

    difference = wide["rmse_mean"] - baseline["rmse_mean"]
    pooled = (wide["rmse_sd"] ** 2 / len(SEEDS) + baseline["rmse_sd"] ** 2 / len(SEEDS)) ** 0.5
    print(f"difference {difference:+,.0f} SGD, standard error of difference {pooled:,.0f}")

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    axes[0].plot(search_frame["lambda"], search_frame["best_validation_rmse"],
                 marker="o", color="#4c78a8")
    axes[0].scatter([selected_lambda], [best["best_validation_rmse"]], s=90,
                    facecolors="none", edgecolors="#d62728", linewidths=1.8,
                    label=f"selected $\\lambda$ = {selected_lambda:g}")
    axes[0].set_xlabel("$\\lambda$")
    axes[0].set_ylabel("Best validation RMSE (SGD)")
    axes[0].set_title("$\\lambda$ selection on 2021 validation", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    labels = ["Baseline (A1)", "Wide-and-deep"]
    colours = ["#4c78a8", "#e45756"]
    for axis, means, sds, ylabel, title in (
        (axes[1], [baseline["rmse_mean"], wide["rmse_mean"]],
         [baseline["rmse_sd"], wide["rmse_sd"]], "Test RMSE (SGD)",
         "Test RMSE (mean $\\pm$ sd, 5 seeds)"),
        (axes[2], [baseline["r2_mean"], wide["r2_mean"]],
         [baseline["r2_sd"], wide["r2_sd"]], "Test $R^2$",
         "Test $R^2$ (mean $\\pm$ sd, 5 seeds)"),
    ):
        axis.bar(labels, means, yerr=sds, capsize=5, color=colours)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3, axis="y")
        axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_a3_wide_and_deep.png", dpi=200)
    plt.close(figure)

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "lambdas": list(LAMBDAS),
                "selected_lambda": selected_lambda,
                "seeds": list(SEEDS),
                "epochs": epochs,
                "lambda_search": search,
                "wide_and_deep": wide,
                "a1_baseline": baseline,
                "rmse_difference": difference,
                "rmse_difference_standard_error": pooled,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
