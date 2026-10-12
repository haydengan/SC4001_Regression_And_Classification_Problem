"""Rebuild the B1 and B2 sweep figures from the saved curves in metrics.json.

Useful for regenerating figures without retraining; b1.py and b2.py produce
identical figures when run end to end.

    uv run python src/b_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = (Path(__file__).resolve().parent.parent / "outputs").resolve()


def plot(curves, key, ylabel, title, path, label, logy=False):
    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    for setting, series in curves.items():
        values = series[key]
        axis.plot(range(1, len(values) + 1), values, marker="o", markersize=3,
                  label=label(setting))
    axis.set_xlabel("Epoch", fontsize=10)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.set_title(title, fontsize=10)
    axis.tick_params(labelsize=9)
    axis.tick_params(axis="x", labelsize=7)
    axis.xaxis.set_major_locator(MultipleLocator(1))
    if logy:
        axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    b1 = json.loads((OUT / "b1" / "metrics.json").read_text())
    b2 = json.loads((OUT / "b2" / "metrics.json").read_text())

    plot(b1["curves"], "loss", "Validation loss",
         "B1: validation loss by learning rate",
         OUT / "b1" / "fig_b1_loss.png", lambda s: f"lr = {float(s):g}", logy=True)
    plot(b1["curves"], "accuracy", "Validation accuracy",
         "B1: validation accuracy by learning rate",
         OUT / "b1" / "fig_b1_accuracy.png", lambda s: f"lr = {float(s):g}")
    plot(b2["curves"], "loss", "Validation loss",
         "B2: validation loss by batch size",
         OUT / "b2" / "fig_b2_loss.png", lambda s: f"batch = {s}", logy=True)
    plot(b2["curves"], "accuracy", "Validation accuracy",
         "B2: validation accuracy by batch size",
         OUT / "b2" / "fig_b2_accuracy.png", lambda s: f"batch = {s}")
    print("rebuilt B1 and B2 sweep figures")


if __name__ == "__main__":
    main()
