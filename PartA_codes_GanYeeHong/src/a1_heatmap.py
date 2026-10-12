"""Heatmap of the A1 grid search, with row and column means.

Reads outputs/a1/grid_search_results.csv and writes
outputs/a1/fig_a1_grid_heatmap.png. Shows the same numbers as the grid table.

    uv run python src/a1_heatmap.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "a1").resolve()
WIDTHS = [16, 32, 64, 128, 256]
EMBEDS = [2, 4, 8, 12, 16]


def main():
    grid = pd.read_csv(OUTPUT_DIR / "grid_search_results.csv")
    pivot = grid.pivot(index="hidden_width", columns="embedding_dim",
                       values="best_validation_rmse").loc[WIDTHS, EMBEDS]
    values = pivot.to_numpy()

    # Grid plus a mean row and a mean column, kept off the colour scale so the
    # 25 configurations keep the full range of shading.
    rows, cols = len(WIDTHS), len(EMBEDS)
    full = np.full((rows + 1, cols + 1), np.nan)
    full[:rows, :cols] = values
    shown = np.ma.masked_invalid(full)

    figure, axis = plt.subplots(figsize=(6.6, 5.0))
    image = axis.imshow(shown, cmap="Blues_r", aspect="auto")

    best = np.unravel_index(np.argmin(values), values.shape)
    for i in range(rows):
        for j in range(cols):
            v = values[i, j]
            mid = (values.max() + values.min()) / 2
            axis.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=9,
                      color="white" if v < mid else "#222222",
                      fontweight="bold" if (i, j) == best else "normal")
    for i in range(rows):
        axis.text(cols, i, f"{values[i].mean():,.0f}", ha="center", va="center",
                  fontsize=9, color="#444444", style="italic")
    for j in range(cols):
        axis.text(j, rows, f"{values[:, j].mean():,.0f}", ha="center", va="center",
                  fontsize=9, color="#444444", style="italic")

    axis.set_xticks(range(cols + 1), [str(e) for e in EMBEDS] + ["mean"])
    axis.set_yticks(range(rows + 1), [str(w) for w in WIDTHS] + ["mean"])
    axis.set_xlabel("Embedding dimension")
    axis.set_ylabel("Hidden layer width")
    axis.set_title("Best validation RMSE (SGD) for each configuration", fontsize=11)
    for edge in ("top", "right", "bottom", "left"):
        axis.spines[edge].set_visible(False)
    axis.tick_params(length=0)

    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    bar.set_label("Validation RMSE (SGD)")
    figure.tight_layout()
    path = OUTPUT_DIR / "fig_a1_grid_heatmap.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"wrote {path}")
    print(f"best cell: width {WIDTHS[best[0]]}, embedding {EMBEDS[best[1]]}, "
          f"{values[best]:,.0f}")


if __name__ == "__main__":
    main()
