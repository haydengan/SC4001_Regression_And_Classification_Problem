"""DIAGNOSTIC ONLY - not part of the submitted Question A1 result.

Question A1 fixes the combined-set retrain to the epoch budget selected on the
2021 validation year, because once validation is folded into training there is
no held-out split left to drive early stopping. This script asks what would
have happened if early stopping had instead been driven by the 2022 test year.

That is test leakage: the test set would be influencing when training stops, so
the resulting number is not an honest estimate of unseen performance. The point
of the script is to measure how large that optimism is.

It trains the selected A1 configuration once on combined train+validation for
the full 20-epoch budget, records test RMSE and R^2 at every epoch, and then
reports three numbers computed from that single run:

  (a) honest      - the submitted protocol: stop at the epoch chosen on validation
  (b) leaky       - early stopping (patience 4, threshold 1e-4) monitored on test
  (c) oracle      - the best test epoch of all 20, i.e. the most optimistic choice

Run from the repository root:

    uv run python src/a1_probe_test_early_stopping.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a1 import DATA_PATH, OUTPUT_DIR, build_model, prepare_combined_data, trial_config
from config import Config
from training import evaluate, train_model


def simulate_early_stopping(rmse_by_epoch, patience, threshold):
    """Replay the assignment's early-stopping rule over a recorded RMSE curve.

    Returns (selected epoch, epoch at which training would have halted).
    """
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch, rmse in enumerate(rmse_by_epoch, start=1):
        if rmse < best_rmse - threshold:
            best_rmse = rmse
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                return best_epoch, epoch
    return best_epoch, len(rmse_by_epoch)


def main():
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text())
    base_config = Config()
    config = trial_config(
        base_config,
        metrics["selected_hidden_width"],
        metrics["selected_embedding_dim"],
    )
    honest_epoch = metrics["selected_best_epoch"]

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    combined = prepare_combined_data(DATA_PATH, config)
    print(
        f"combined train {len(combined['train']):,} | test {len(combined['test']):,}"
    )
    print(
        f"selected configuration: width {config.hidden_width}, "
        f"embedding {config.embedding_dims[0]}, seed {config.seed}\n"
    )

    # Train the full 20-epoch budget with no early stopping, recording the test
    # metrics after every epoch. Test never influences the weights here; it is
    # only observed, which lets all three protocols be scored from one run.
    model = build_model(combined, config)
    test_rmse_by_epoch = []
    test_r2_by_epoch = []

    def record(epoch, train_rmse, _unused):
        rmse, r_squared = evaluate(model, combined["test"], config.batch_size)
        test_rmse_by_epoch.append(rmse)
        test_r2_by_epoch.append(r_squared)
        print(
            f"epoch {epoch:02d} | train RMSE {train_rmse:,.0f} "
            f"| test RMSE {rmse:,.0f} | test R2 {r_squared:.4f}"
        )

    history = train_model(
        model,
        combined["train"],
        config,
        evaluation_data=None,
        max_epochs=config.max_epochs,
        early_stopping=False,
        on_epoch_end=record,
        verbose=False,
    )

    leaky_epoch, halt_epoch = simulate_early_stopping(
        test_rmse_by_epoch,
        base_config.early_stopping_patience,
        base_config.early_stopping_threshold,
    )
    oracle_epoch = min(
        range(1, len(test_rmse_by_epoch) + 1), key=lambda e: test_rmse_by_epoch[e - 1]
    )

    rows = [
        (
            "(a) honest: stop at epoch chosen on 2021 validation",
            honest_epoch,
            test_rmse_by_epoch[honest_epoch - 1],
            test_r2_by_epoch[honest_epoch - 1],
        ),
        (
            f"(b) leaky: early stopping monitored on test (halts epoch {halt_epoch})",
            leaky_epoch,
            test_rmse_by_epoch[leaky_epoch - 1],
            test_r2_by_epoch[leaky_epoch - 1],
        ),
        (
            "(c) oracle: best test epoch of all 20",
            oracle_epoch,
            test_rmse_by_epoch[oracle_epoch - 1],
            test_r2_by_epoch[oracle_epoch - 1],
        ),
    ]

    print("\n--- protocol comparison (same single run, seed 42) ---")
    print(f"{'protocol':<58}{'epoch':>6}{'test RMSE':>13}{'test R2':>10}")
    for label, epoch, rmse, r_squared in rows:
        print(f"{label:<58}{epoch:>6}{rmse:>13,.0f}{r_squared:>10.4f}")

    honest_rmse = rows[0][2]
    print(
        f"\noptimism from leaking test into early stopping: "
        f"SGD {honest_rmse - rows[1][2]:,.0f} RMSE "
        f"({100 * (honest_rmse - rows[1][2]) / honest_rmse:.2f}% of the honest RMSE), "
        f"R2 +{rows[1][3] - rows[0][3]:.4f}"
    )
    print(
        f"optimism from the oracle choice:                 "
        f"SGD {honest_rmse - rows[2][2]:,.0f} RMSE "
        f"({100 * (honest_rmse - rows[2][2]) / honest_rmse:.2f}% of the honest RMSE), "
        f"R2 +{rows[2][3] - rows[0][3]:.4f}"
    )
    print(
        f"\nspread of test RMSE across all 20 epochs: "
        f"SGD {min(test_rmse_by_epoch):,.0f} to {max(test_rmse_by_epoch):,.0f}"
    )

    epochs = list(range(1, len(test_rmse_by_epoch) + 1))
    frame = pd.DataFrame(
        {
            "epoch": epochs,
            "train_rmse": history["train_rmse"],
            "test_rmse": test_rmse_by_epoch,
            "test_r2": test_r2_by_epoch,
        }
    )
    frame.to_csv(OUTPUT_DIR / "probe_test_early_stopping.csv", index=False)

    figure, axes = plt.subplots(figsize=(6.5, 4.0))
    axes.plot(epochs, test_rmse_by_epoch, marker="o", markersize=3.5,
              color="#ff7f0e", label="Test RMSE (2022)")
    for epoch, colour, label in (
        (honest_epoch, "#1f77b4", f"(a) honest, epoch {honest_epoch}"),
        (leaky_epoch, "#d62728", f"(b) leaky, epoch {leaky_epoch}"),
        (oracle_epoch, "#2ca02c", f"(c) oracle, epoch {oracle_epoch}"),
    ):
        axes.axvline(epoch, linestyle="--", linewidth=1.1, color=colour, label=label)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Test RMSE (SGD)")
    axes.set_title("Diagnostic: which epoch each stopping protocol would pick")
    axes.set_xticks(epochs)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_probe_test_early_stopping.png", dpi=200)
    plt.close(figure)

    print(f"\nwrote {OUTPUT_DIR / 'probe_test_early_stopping.csv'}")
    print(f"wrote {OUTPUT_DIR / 'fig_probe_test_early_stopping.png'}")


if __name__ == "__main__":
    main()
