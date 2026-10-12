"""Train and validation curves for the three selected architectures.

Each network is trained at its selected hyperparameters with 2021 held out, so
the curves show convergence and the train-to-validation gap. These are the
learning curves only; the reported test metrics come from b1.py to b3.py.

    uv run python src/b_curves.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_data
from models import AlexNetLike, ResNetLike, VGGLike
from training import train_model

OUT = (SRC_DIR.parent / "outputs").resolve()
SEED = 42


def main():
    lr = json.loads((OUT / "b1" / "metrics.json").read_text())["selected_learning_rate"]
    batch = json.loads((OUT / "b2" / "metrics.json").read_text())["selected_batch_size"]
    data = load_data()

    settings = [
        ("B1 AlexNet-like", AlexNetLike, 128),
        ("B2 VGG-like", VGGLike, batch),
        ("B3 ResNet-like", ResNetLike, batch),
    ]
    curves = {}
    for name, cls, bs in settings:
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        history = train_model(cls(), data["train"], data["validation"], lr, bs)
        curves[name] = history
        print(f"{name:<18} batch {bs:<4} stopped after "
              f"{len(history['train_loss'])} epochs, best epoch {history['best_epoch']}")

    figure, axes = plt.subplots(2, 3, figsize=(10.5, 5.2))
    for column, (name, _cls, bs) in enumerate(settings):
        h = curves[name]
        epochs = range(1, len(h["train_loss"]) + 1)
        top, bottom = axes[0][column], axes[1][column]
        top.plot(epochs, h["train_loss"], marker="o", markersize=3,
                 color="#4c78a8", label="Train loss")
        top.plot(epochs, [m["loss"] for m in h["validation"]], marker="o",
                 markersize=3, color="#f58518", label="Validation loss")
        top.set_title(f"{name}  (lr {lr:g}, batch {bs})", fontsize=9)
        top.set_ylabel("Loss")
        top.legend(fontsize=7.5)
        bottom.plot(epochs, [m["accuracy"] for m in h["validation"]], marker="o",
                    markersize=3, color="#54a24b", label="Validation accuracy")
        bottom.set_ylabel("Validation accuracy")
        bottom.legend(fontsize=7.5)
        for axis in (top, bottom):
            axis.set_xlabel("Epoch")
            axis.xaxis.set_major_locator(MultipleLocator(1))
            axis.tick_params(labelsize=7.5)
            axis.grid(alpha=0.3)
    figure.tight_layout()
    path = OUT / "b3" / "fig_b_learning_curves.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
