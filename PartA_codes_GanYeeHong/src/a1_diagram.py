"""Draw the block diagram of the baseline model for Question A1.

Run from the repository root:

    uv run python src/a1_diagram.py

Writes outputs/a1/fig_a1_block_diagram.png. Shapes and parameter counts match
the provided baseline (embedding dimension 8, hidden width 64).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "a1").resolve()

INPUT_COLOUR = "#dce9f7"
EMBEDDING_COLOUR = "#f7e2c8"
LAYER_COLOUR = "#d9ead3"
OUTPUT_COLOUR = "#f4cccc"

CATEGORICAL = [
    ("month", 13),
    ("town", 27),
    ("flat_model_type", 44),
    ("storey_range", 18),
]


def box(axes, x, y, width, height, text, colour, fontsize=8.0):
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=colour,
            edgecolor="#444444",
            linewidth=0.9,
        )
    )
    axes.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.35,
    )


def arrow(axes, start, end):
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color="#444444",
            shrinkA=0,
            shrinkB=0,
        )
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(9.0, 6.0))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 8.2)
    axes.axis("off")

    # Row 1: categorical inputs and their embedding tables.
    width = 1.85
    gap = 0.22
    left = 0.35
    for index, (name, cardinality) in enumerate(CATEGORICAL):
        x = left + index * (width + gap)
        box(
            axes,
            x,
            7.3,
            width,
            0.70,
            f"{name}\n(categorical)",
            INPUT_COLOUR,
        )
        box(
            axes,
            x,
            6.1,
            width,
            0.80,
            f"nn.Embedding\n({cardinality}, 8)\n{cardinality * 8} params",
            EMBEDDING_COLOUR,
            fontsize=8.4,
        )
        arrow(axes, (x + width / 2, 7.3), (x + width / 2, 6.95))
        arrow(axes, (x + width / 2, 6.1), (x + width / 2, 5.80))

    # Row 1 (right): the six continuous features bypass the embedding row.
    box(
        axes,
        8.6,
        6.1,
        1.1,
        1.90,
        "6 continuous\nfeatures\n\nstandardised\n(train mean/std)\n\n0 params",
        INPUT_COLOUR,
        fontsize=8.0,
    )
    arrow(axes, (9.15, 6.1), (9.15, 5.80))

    # Concatenation.
    box(
        axes,
        0.35,
        5.15,
        9.35,
        0.65,
        "Concatenate  ->  4 x 8 embedding dims + 6 continuous = 38 features",
        LAYER_COLOUR,
        fontsize=9.6,
    )
    arrow(axes, (5.0, 5.15), (5.0, 4.80))

    centre_x = 2.9
    centre_width = 4.2
    stack = [
        (4.15, "Linear (38 -> 64)\nweights 38x64 + bias 64  =  2,496 params", LAYER_COLOUR),
        (3.20, "LayerNorm (64)\nscale 64 + shift 64  =  128 params", LAYER_COLOUR),
        (2.25, "ReLU\n0 params", LAYER_COLOUR),
        (1.30, "Linear (64 -> 1)\nweights 64 + bias 1  =  65 params", LAYER_COLOUR),
    ]
    for y, text, colour in stack:
        box(axes, centre_x, y, centre_width, 0.65, text, colour, fontsize=9.2)
    arrow(axes, (5.0, 4.15), (5.0, 3.85))
    arrow(axes, (5.0, 3.20), (5.0, 2.90))
    arrow(axes, (5.0, 2.25), (5.0, 1.95))
    arrow(axes, (5.0, 1.30), (5.0, 1.00))

    box(
        axes,
        centre_x,
        0.35,
        centre_width,
        0.65,
        "Predicted resale price (SGD)\nprice = output x target std + target mean",
        OUTPUT_COLOUR,
        fontsize=9.2,
    )

    axes.text(
        5.0,
        0.08,
        "Total trainable parameters: 816 (embeddings) + 2,496 + 128 + 65 = 3,505",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
    )

    figure.tight_layout()
    path = OUTPUT_DIR / "fig_a1_block_diagram.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
