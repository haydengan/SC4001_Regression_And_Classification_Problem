"""Question B2: VGG-like network, batch-size search.

Uses the learning rate selected in B1. The optimal batch size is the one with
the lowest validation classification error at its best epoch; that model is
then evaluated once on the test set.

    uv run python src/b2.py
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
from models import VGGLike, count_flops, count_parameters
from training import evaluate, train_model

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "b2").resolve()
B1_METRICS = (SRC_DIR.parent / "outputs" / "b1" / "metrics.json").resolve()
BATCH_SIZES = (16, 32, 64, 128, 256)
SEED = 42


def plot_series(histories, key, ylabel, title, path, logy=False):
    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    for batch_size, history in histories.items():
        values = [m[key] for m in history["validation"]]
        axis.plot(range(1, len(values) + 1), values, marker="o", markersize=3,
                  label=f"batch = {batch_size}")
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
    learning_rate = json.loads(B1_METRICS.read_text())["selected_learning_rate"]
    print(f"learning rate from B1: {learning_rate:g}\n")
    data = load_data()

    histories = {}
    records = []
    models = {}
    for batch_size in BATCH_SIZES:
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = VGGLike()
        history = train_model(
            model, data["train"], data["validation"], learning_rate, batch_size
        )
        best = history["validation"][history["best_epoch"] - 1]
        histories[batch_size] = history
        models[batch_size] = model
        records.append(
            {
                "batch_size": batch_size,
                "best_epoch": history["best_epoch"],
                "epochs_run": history["epochs_run"],
                **{k: best[k] for k in
                   ("loss", "accuracy", "error", "sensitivity", "specificity")},
            }
        )
        print(
            f"batch {batch_size:<4d} | best epoch {history['best_epoch']:2d} "
            f"| val loss {best['loss']:.4f} | val error {best['error']:.4f} "
            f"| sens {best['sensitivity']:.4f} | spec {best['specificity']:.4f}"
        )

    selected = min(records, key=lambda r: (r["error"], r["loss"]))
    selected_batch = selected["batch_size"]
    print(f"\nselected batch size: {selected_batch}")

    selected_epochs = selected["best_epoch"]
    combined = combined_training_set(data)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = VGGLike()
    train_model(model, combined, None, learning_rate, selected_batch,
                max_epochs=selected_epochs, early_stopping=False)
    test = evaluate(model, data["test"])
    print(f"retrained on train+validation ({len(combined):,} images) "
          f"for {selected_epochs} epochs")
    print(
        f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} "
        f"| specificity {test['specificity']:.4f}"
    )

    torch.save(model.state_dict(), OUTPUT_DIR / "b2_model.pt")
    plot_series(histories, "loss", "Validation loss",
                "B2: validation loss by batch size",
                OUTPUT_DIR / "fig_b2_loss.png", logy=True)
    plot_series(histories, "accuracy", "Validation accuracy",
                "B2: validation accuracy by batch size",
                OUTPUT_DIR / "fig_b2_accuracy.png")

    reference = VGGLike()
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "learning_rate": learning_rate,
                "batch_sizes": list(BATCH_SIZES),
                "search": records,
                "selected_batch_size": selected_batch,
                "retrain_epochs": selected["best_epoch"],
                "test": test,
                "trainable_parameters": count_parameters(reference),
                "flops": count_flops(reference),
                "curves": {
                    str(b): {
                        "loss": [m["loss"] for m in h["validation"]],
                        "accuracy": [m["accuracy"] for m in h["validation"]],
                    }
                    for b, h in histories.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
