"""Question A1: parameter count, Ray grid search, and combined-set retrain.

Run from the repository root:

    uv run python src/a1.py

Pipeline
--------
1. Report the per-layer trainable parameter count of the provided baseline
   (embedding dimension 8, hidden width 64).
2. Grid search hidden width x embedding dimension with Ray Tune, training on
   2017-2020 and selecting on the 2021 validation year.
3. Select the configuration with the lowest best validation RMSE and plot its
   training and validation RMSE curves.
4. Refit preprocessing on train+validation, retrain the selected configuration
   on those rows for the selected number of epochs, and evaluate once on the
   2022 test year. Plot training and test RMSE curves, and report RMSE and R^2.

Outputs are written to outputs/a1/.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import torch
from ray import tune

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config
from data import make_dataset, prepare_data
from model import PriceModel
from training import (
    evaluate,
    parameter_breakdown,
    residual_diagnostics,
    train_model,
)

DATA_PATH = (SRC_DIR.parent / "hdb_price_prediction.csv").resolve()
OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "a1").resolve()

# A1 search space: hidden-layer width x one shared embedding dimension that is
# applied to all four categorical features.
HIDDEN_WIDTHS = (16, 32, 64, 128, 256)
EMBEDDING_DIMS = (2, 4, 8, 12, 16)

READABLE_PARAMETER_NAMES = {
    "embeddings.0.weight": "Embedding: month",
    "embeddings.1.weight": "Embedding: town",
    "embeddings.2.weight": "Embedding: flat_model_type",
    "embeddings.3.weight": "Embedding: storey_range",
    "mlp.0.weight": "Linear 1 weight",
    "mlp.0.bias": "Linear 1 bias",
    "mlp.1.weight": "LayerNorm scale",
    "mlp.1.bias": "LayerNorm shift",
    "mlp.3.weight": "Linear 2 weight",
    "mlp.3.bias": "Linear 2 bias",
}


def trial_config(config: Config, hidden_width: int, embedding_dim: int) -> Config:
    """Baseline config with one grid point applied to all categorical features."""
    return replace(
        config,
        hidden_width=hidden_width,
        embedding_dims=(embedding_dim,) * len(config.categorical_features),
    )


def build_model(data, config: Config) -> PriceModel:
    return PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )


def prepare_combined_data(csv_path, config: Config):
    """Refit preprocessing on train+validation and return those rows plus test.

    Mirrors ``data.prepare_data`` but treats 2017-2021 as the training set. The
    category encoders and the standardisation statistics are therefore fitted on
    train+validation: preprocessing statistics must always come from the rows
    the model is trained on, and never from the test year.
    """
    rows = pd.read_csv(csv_path).to_dict(orient="records")
    combined_years = set(config.train_years) | {config.validation_year}
    combined_rows = [row for row in rows if int(row["year"]) in combined_years]
    test_rows = [row for row in rows if int(row["year"]) == config.test_year]

    category_maps = {}
    for name in config.categorical_features:
        values = sorted({row[name] for row in combined_rows})
        category_maps[name] = {value: index + 1 for index, value in enumerate(values)}

    combined_continuous = torch.tensor(
        [
            [float(row[name]) for name in config.continuous_features]
            for row in combined_rows
        ]
    )
    means = combined_continuous.mean(dim=0)
    stds = combined_continuous.std(dim=0).clamp_min(1e-8)

    combined_target = torch.tensor([float(row["resale_price"]) for row in combined_rows])

    return {
        "train": make_dataset(combined_rows, category_maps, means, stds, config),
        "test": make_dataset(test_rows, category_maps, means, stds, config),
        "cardinalities": [
            len(category_maps[name]) + 1 for name in config.categorical_features
        ],
        "target_mean": combined_target.mean().item(),
        "target_std": combined_target.std(correction=0).clamp_min(1e-8).item(),
    }


def tune_trainable(search_space, data=None, base_config=None):
    """One Ray Tune trial: train a single grid point and report each epoch."""
    torch.set_num_threads(1)
    config = trial_config(
        base_config, search_space["hidden_width"], search_space["embedding_dim"]
    )
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = build_model(data, config)
    _, total_parameters = parameter_breakdown(model)

    def report_epoch(epoch, train_rmse, validation_rmse):
        tune.report(
            {
                "epoch": epoch,
                "train_rmse": train_rmse,
                "validation_rmse": validation_rmse,
                "trainable_parameters": total_parameters,
            }
        )

    train_model(
        model,
        data["train"],
        config,
        evaluation_data=data["validation"],
        early_stopping=True,
        on_epoch_end=report_epoch,
        verbose=False,
    )


def run_grid_search(data, base_config: Config):
    """Run the 5x5 grid with Ray Tune and return (results table, curves)."""
    import ray

    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        configure_logging=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(SRC_DIR)}},
    )

    trainable = tune.with_parameters(
        tune_trainable, data=data, base_config=base_config
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 1}),
        param_space={
            "hidden_width": tune.grid_search(list(HIDDEN_WIDTHS)),
            "embedding_dim": tune.grid_search(list(EMBEDDING_DIMS)),
        },
        run_config=tune.RunConfig(
            name="a1_grid_search",
            storage_path=str(OUTPUT_DIR / "ray"),
            verbose=1,
        ),
    )
    results = tuner.fit()

    records = []
    curves = {}
    for result in results:
        frame = result.metrics_dataframe
        best_row = frame.loc[frame["validation_rmse"].idxmin()]
        key = (result.config["hidden_width"], result.config["embedding_dim"])
        records.append(
            {
                "hidden_width": key[0],
                "embedding_dim": key[1],
                "trainable_parameters": int(best_row["trainable_parameters"]),
                "epochs_run": int(frame["epoch"].max()),
                "best_epoch": int(best_row["epoch"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
            }
        )
        curves[key] = frame[["epoch", "train_rmse", "validation_rmse"]].copy()

    ray.shutdown()
    table = pd.DataFrame(records).sort_values("best_validation_rmse").reset_index(drop=True)
    return table, curves


def annotate_extremes(panel, xs, ys, colour):
    """Ring and label the highest and lowest point of a curve."""
    xs, ys = list(xs), list(ys)
    lowest = min(range(len(ys)), key=lambda i: ys[i])
    highest = max(range(len(ys)), key=lambda i: ys[i])
    for index, offset, va in ((highest, 10, "bottom"), (lowest, -12, "top")):
        # Nudge labels inward at the ends so they do not run off the panel.
        if index == 0:
            shift, align = 5, "left"
        elif index == len(xs) - 1:
            shift, align = -5, "right"
        else:
            shift, align = 0, "center"
        panel.scatter([xs[index]], [ys[index]], s=42, facecolors="none",
                      edgecolors=colour, linewidths=1.4, zorder=5)
        panel.annotate(f"{ys[index]:,.0f}", (xs[index], ys[index]),
                       textcoords="offset points", xytext=(shift, offset), ha=align,
                       va=va, fontsize=7.5, color=colour, fontweight="bold", zorder=6)


def plot_curves(epochs, series, labels, title, path):
    """Two RMSE curves on the same axes."""
    figure, axes = plt.subplots(figsize=(6.5, 4.0))
    for values, label in zip(series, labels):
        line, = axes.plot(epochs, values, marker="o", markersize=3.5, label=label)
        annotate_extremes(axes, epochs, values, line.get_color())
    axes.set_xlabel("Epoch")
    axes.set_ylabel("RMSE (SGD)")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)
    axes.margins(x=0.05, y=0.18)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = Config()
    torch.manual_seed(base_config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("preparing data (train 2017-2020 / validation 2021 / test 2022)")
    data = prepare_data(DATA_PATH, base_config)
    print(
        f"  train {len(data['train']):,} | validation {len(data['validation']):,} "
        f"| test {len(data['test']):,}"
    )
    print(f"  embedding cardinalities {data['cardinalities']}")

    # ---------------------------------------------------------------- step 1
    baseline_model = build_model(data, base_config)
    rows, baseline_total = parameter_breakdown(baseline_model)
    lines = [
        "Baseline PriceModel trainable parameters "
        f"(embedding dim {base_config.embedding_dims[0]}, hidden width {base_config.hidden_width})",
        "",
        f"{'Block':<28}{'Shape':<14}{'Parameters':>12}",
    ]
    for name, shape, count in rows:
        label = READABLE_PARAMETER_NAMES.get(name, name)
        lines.append(f"{label:<28}{str(tuple(shape)):<14}{count:>12,}")
    lines.append(f"{'Total':<28}{'':<14}{baseline_total:>12,}")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    (OUTPUT_DIR / "parameter_count.txt").write_text(report + "\n")

    # ---------------------------------------------------------------- step 2
    print(f"grid search: {len(HIDDEN_WIDTHS)} widths x {len(EMBEDDING_DIMS)} embedding dims")
    table, curves = run_grid_search(data, base_config)
    table.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)

    pivot = table.pivot(
        index="hidden_width", columns="embedding_dim", values="best_validation_rmse"
    )
    pivot_text = (
        "Best validation RMSE (SGD) by hidden width (rows) x embedding dimension (columns)\n\n"
        + pivot.round(0).to_string()
    )
    print("\n" + pivot_text + "\n")
    (OUTPUT_DIR / "grid_search_table.txt").write_text(pivot_text + "\n")
    print(table.to_string(index=False))

    best = table.iloc[0]
    selected_width = int(best["hidden_width"])
    selected_embedding = int(best["embedding_dim"])
    selected_epochs = int(best["best_epoch"])
    print(
        f"\nselected: hidden width {selected_width}, embedding dim {selected_embedding}"
        f" | best validation RMSE {best['best_validation_rmse']:,.0f}"
        f" at epoch {selected_epochs}"
    )

    # ---------------------------------------------------------------- step 3
    selected_curve = curves[(selected_width, selected_embedding)]
    plot_curves(
        selected_curve["epoch"].tolist(),
        [selected_curve["train_rmse"].tolist(), selected_curve["validation_rmse"].tolist()],
        ["Training RMSE (2017-2020)", "Validation RMSE (2021)"],
        f"Selected configuration: width {selected_width}, embedding {selected_embedding}",
        OUTPUT_DIR / "fig_a1_train_validation.png",
    )
    selected_curve.to_csv(OUTPUT_DIR / "curve_train_validation.csv", index=False)

    # ---------------------------------------------------------------- step 4
    print("\nretraining on combined train+validation (2017-2021), preprocessing refitted")
    combined = prepare_combined_data(DATA_PATH, base_config)
    print(f"  combined train {len(combined['train']):,} | test {len(combined['test']):,}")

    selected_config = trial_config(base_config, selected_width, selected_embedding)
    torch.manual_seed(selected_config.seed)
    final_model = build_model(combined, selected_config)
    _, selected_total = parameter_breakdown(final_model)

    history = train_model(
        final_model,
        combined["train"],
        selected_config,
        evaluation_data=combined["test"],
        max_epochs=selected_epochs,
        early_stopping=False,
        verbose=True,
    )

    epochs = list(range(1, len(history["train_rmse"]) + 1))
    plot_curves(
        epochs,
        [history["train_rmse"], history["evaluation_rmse"]],
        ["Training RMSE (2017-2021)", "Test RMSE (2022)"],
        f"Combined-set retrain: width {selected_width}, embedding {selected_embedding}",
        OUTPUT_DIR / "fig_a1_train_test.png",
    )
    pd.DataFrame(
        {
            "epoch": epochs,
            "train_rmse": history["train_rmse"],
            "test_rmse": history["evaluation_rmse"],
        }
    ).to_csv(OUTPUT_DIR / "curve_train_test.csv", index=False)

    # Both A1 curve requirements as one two-panel figure, to save report space.
    figure, panels = plt.subplots(1, 2, figsize=(9.2, 3.5))
    left = [
        (selected_curve["epoch"], selected_curve["train_rmse"], "Training (2017-2020)"),
        (selected_curve["epoch"], selected_curve["validation_rmse"], "Validation (2021)"),
    ]
    right = [
        (epochs, history["train_rmse"], "Training (2017-2021)"),
        (epochs, history["evaluation_rmse"], "Test (2022)"),
    ]
    for panel, series, title in (
        (panels[0], left, "Model selection: train vs validation"),
        (panels[1], right, "Combined-set retrain: train vs test"),
    ):
        for xs, ys, label in series:
            line, = panel.plot(xs, ys, marker="o", markersize=3, label=label)
            annotate_extremes(panel, xs, ys, line.get_color())
        panel.set_title(title, fontsize=10)
        panel.set_xlabel("Epoch")
        panel.set_ylabel("RMSE (SGD)")
        panel.xaxis.set_major_locator(MaxNLocator(integer=True))
        panel.margins(x=0.05, y=0.18)
        panel.grid(alpha=0.3)
        panel.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_a1_curves.png", dpi=200)
    plt.close(figure)

    train_rmse, train_r2 = evaluate(
        final_model, combined["train"], selected_config.batch_size
    )
    test_rmse, test_r2 = evaluate(
        final_model, combined["test"], selected_config.batch_size
    )

    # Bias / spread decomposition supporting the underfitting-vs-overfitting
    # discussion: a large bias share points to distribution shift across years
    # rather than to the model memorising its training set.
    train_residuals = residual_diagnostics(
        final_model, combined["train"], selected_config.batch_size
    )
    test_residuals = residual_diagnostics(
        final_model, combined["test"], selected_config.batch_size
    )

    torch.save(final_model.state_dict(), OUTPUT_DIR / "a1_final_model.pt")
    metrics = {
        "baseline_trainable_parameters": baseline_total,
        "selected_hidden_width": selected_width,
        "selected_embedding_dim": selected_embedding,
        "selected_trainable_parameters": selected_total,
        "selected_best_epoch": selected_epochs,
        "selection_best_validation_rmse": float(best["best_validation_rmse"]),
        "combined_train_rmse": train_rmse,
        "combined_train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "train_residuals": train_residuals,
        "test_residuals": test_residuals,
        "seed": selected_config.seed,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n--- A1 final results ---")
    print(f"selected hidden width        : {selected_width}")
    print(f"selected embedding dimension : {selected_embedding}")
    print(f"trainable parameters         : {selected_total:,}")
    print(f"epochs for combined retrain  : {selected_epochs}")
    print(f"train RMSE (2017-2021)       : {train_rmse:,.0f}   R^2 {train_r2:.4f}")
    print(f"test  RMSE (2022)            : {test_rmse:,.0f}   R^2 {test_r2:.4f}")
    print("\nresidual decomposition (supports the A1 discussion)")
    for name, diagnostics in (("train", train_residuals), ("test", test_residuals)):
        print(
            f"  {name:<6} mean actual {diagnostics['mean_actual']:>9,.0f}"
            f" | bias {diagnostics['bias']:>9,.0f}"
            f" | residual sd {diagnostics['residual_sd']:>8,.0f}"
            f" | bias share of MSE {diagnostics['bias_share_of_mse']:.3f}"
        )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
