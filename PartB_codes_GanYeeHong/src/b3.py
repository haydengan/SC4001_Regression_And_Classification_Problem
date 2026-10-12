"""Question B3: ResNet-like network and the B1-B3 comparison.

Uses the learning rate from B1 and the batch size from B2, then tabulates all
three architectures on test accuracy, sensitivity, specificity, classification
error, learnable parameters and FLOPs.

    uv run python src/b3.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import combined_training_set, load_data
from models import ResNetLike, count_flops, count_parameters
from training import evaluate, train_model

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "b3").resolve()
B1_METRICS = (SRC_DIR.parent / "outputs" / "b1" / "metrics.json").resolve()
B2_METRICS = (SRC_DIR.parent / "outputs" / "b2" / "metrics.json").resolve()
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b1 = json.loads(B1_METRICS.read_text())
    b2 = json.loads(B2_METRICS.read_text())
    learning_rate = b1["selected_learning_rate"]
    batch_size = b2["selected_batch_size"]
    print(f"learning rate {learning_rate:g} (B1), batch size {batch_size} (B2)\n")

    data = load_data()
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = ResNetLike()
    history = train_model(
        model, data["train"], data["validation"], learning_rate, batch_size,
        verbose=True,
    )
    validation = history["validation"][history["best_epoch"] - 1]

    # Final model: fresh retrain on train+validation for the selected epochs.
    combined = combined_training_set(data)
    torch.manual_seed(SEED)
    model = ResNetLike()
    train_model(model, combined, None, learning_rate, batch_size,
                max_epochs=history["best_epoch"], early_stopping=False)
    test = evaluate(model, data["test"])
    print(f"\nretrained on train+validation ({len(combined):,} images) "
          f"for {history['best_epoch']} epochs")
    print(
        f"\nB3 best epoch {history['best_epoch']} | val error {validation['error']:.4f}"
        f"\ntest accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} "
        f"| specificity {test['specificity']:.4f}"
    )
    torch.save(model.state_dict(), OUTPUT_DIR / "b3_model.pt")

    comparison = [
        {
            "architecture": "B1 AlexNet-like",
            "setting": f"lr {learning_rate:g}, batch {b1['batch_size']}",
            **b1["test"],
            "trainable_parameters": b1["trainable_parameters"],
            "flops": b1["flops"],
        },
        {
            "architecture": "B2 VGG-like",
            "setting": f"lr {learning_rate:g}, batch {batch_size}",
            **b2["test"],
            "trainable_parameters": b2["trainable_parameters"],
            "flops": b2["flops"],
        },
        {
            "architecture": "B3 ResNet-like",
            "setting": f"lr {learning_rate:g}, batch {batch_size}",
            **test,
            "trainable_parameters": count_parameters(ResNetLike()),
            "flops": count_flops(ResNetLike()),
        },
    ]

    print(f"\n{'architecture':<18}{'acc':>8}{'sens':>8}{'spec':>8}{'error':>8}"
          f"{'params':>10}{'MFLOPs':>9}")
    for row in comparison:
        print(
            f"{row['architecture']:<18}{row['accuracy']:>8.4f}{row['sensitivity']:>8.4f}"
            f"{row['specificity']:>8.4f}{row['error']:>8.4f}"
            f"{row['trainable_parameters']:>10,}{row['flops'] / 1e6:>9.2f}"
        )

    names = [row["architecture"] for row in comparison]
    colours = ["#4c78a8", "#f58518", "#54a24b"]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for axis, (key, label) in zip(
        axes,
        [("accuracy", "Test accuracy"), ("sensitivity", "Test sensitivity"),
         ("specificity", "Test specificity")],
    ):
        axis.bar(names, [row[key] for row in comparison], color=colours)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(label)
        axis.set_title(label, fontsize=10)
        axis.tick_params(axis="x", labelsize=7.5, rotation=12)
        axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_b3_comparison.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    for row, colour in zip(comparison, colours):
        axis.scatter(row["flops"] / 1e6, row["accuracy"], s=90, color=colour,
                     label=row["architecture"])
        axis.annotate(f"{row['trainable_parameters'] / 1e3:.0f}k params",
                      (row["flops"] / 1e6, row["accuracy"]),
                      textcoords="offset points", xytext=(0, -15),
                      ha="center", fontsize=7.5)
    axis.margins(x=0.22, y=0.26)
    axis.set_xlabel("Forward-pass FLOPs per image (millions)")
    axis.set_ylabel("Test accuracy")
    axis.set_title("Cost against accuracy", fontsize=10)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_b3_cost.png", dpi=200)
    plt.close(figure)

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "best_epoch": history["best_epoch"],
                "validation": validation,
                "test": test,
                "comparison": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
