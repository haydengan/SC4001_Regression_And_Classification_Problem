"""Question B1: AlexNet-like network, learning-rate search.

Batch size 128, learning rates {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}. The optimal
learning rate is the one with the lowest validation classification error at
its best epoch; that model is then evaluated once on the test set.

    uv run python src/b1.py
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

from data import combined_training_set, load_data
from models import AlexNetLike, count_flops, count_parameters
from training import evaluate, train_model

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "b1").resolve()
LEARNING_RATES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
BATCH_SIZE = 128
SEED = 42


def plot_series(histories, key, ylabel, title, path, logy=False):
    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    for learning_rate, history in histories.items():
        values = [m[key] for m in history["validation"]]
        axis.plot(range(1, len(values) + 1), values, marker="o", markersize=3,
                  label=f"lr = {learning_rate:g}")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    histories = {}
    records = []
    models = {}
    for learning_rate in LEARNING_RATES:
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = AlexNetLike()
        history = train_model(
            model, data["train"], data["validation"], learning_rate, BATCH_SIZE
        )
        best = history["validation"][history["best_epoch"] - 1]
        histories[learning_rate] = history
        models[learning_rate] = model
        records.append(
            {
                "learning_rate": learning_rate,
                "best_epoch": history["best_epoch"],
                "epochs_run": history["epochs_run"],
                **{k: best[k] for k in
                   ("loss", "accuracy", "error", "sensitivity", "specificity")},
            }
        )
        print(
            f"lr {learning_rate:<8g} | best epoch {history['best_epoch']:2d} "
            f"| val loss {best['loss']:.4f} | val error {best['error']:.4f} "
            f"| sens {best['sensitivity']:.4f} | spec {best['specificity']:.4f}"
        )

    selected = min(records, key=lambda r: (r["error"], r["loss"]))
    selected_lr = selected["learning_rate"]
    print(f"\nselected learning rate: {selected_lr:g}")

    # Final model: a fresh network retrained from scratch on train+validation
    # for the epoch count found in the search, then tested once.
    selected_epochs = selected["best_epoch"]
    combined = combined_training_set(data)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = AlexNetLike()
    train_model(model, combined, None, selected_lr, BATCH_SIZE,
                max_epochs=selected_epochs, early_stopping=False)
    test = evaluate(model, data["test"])
    print(f"retrained on train+validation ({len(combined):,} images) "
          f"for {selected_epochs} epochs")
    print(
        f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} "
        f"| specificity {test['specificity']:.4f}"
    )

    torch.save(model.state_dict(), OUTPUT_DIR / "b1_model.pt")
    plot_series(histories, "loss", "Validation loss",
                "B1: validation loss by learning rate",
                OUTPUT_DIR / "fig_b1_loss.png", logy=True)
    plot_series(histories, "accuracy", "Validation accuracy",
                "B1: validation accuracy by learning rate",
                OUTPUT_DIR / "fig_b1_accuracy.png")

    reference = AlexNetLike()
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "learning_rates": list(LEARNING_RATES),
                "search": records,
                "selected_learning_rate": selected_lr,
                "retrain_epochs": selected["best_epoch"],
                "retrain_images": len(combined_training_set(data)),
                "test": test,
                "trainable_parameters": count_parameters(reference),
                "flops": count_flops(reference),
                "curves": {
                    str(lr): {
                        "loss": [m["loss"] for m in h["validation"]],
                        "accuracy": [m["accuracy"] for m in h["validation"]],
                    }
                    for lr, h in histories.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
