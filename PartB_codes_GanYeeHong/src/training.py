"""Training loop and binary classification metrics for Part B.

Adam, at most 20 epochs, early stopping with patience 4 and minimum
improvement 1e-4 on validation loss, as specified in the assignment brief.
Class 1 is pneumonia, so sensitivity is recall on pneumonia and specificity is
recall on the healthy class.
"""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

MAX_EPOCHS = 20
PATIENCE = 4
MIN_IMPROVEMENT = 1e-4


@torch.no_grad()
def evaluate(model, dataset, batch_size=256):
    """Return loss, accuracy, error, sensitivity and specificity."""
    model.eval()
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    logits = []
    targets = []
    for images, labels in DataLoader(dataset, batch_size=batch_size):
        output = model(images)
        total_loss += loss_function(output, labels).item()
        logits.append(output)
        targets.append(labels)

    logit = torch.cat(logits)
    target = torch.cat(targets)
    prediction = (torch.sigmoid(logit) >= 0.5).float()

    true_positive = ((prediction == 1) & (target == 1)).sum().item()
    true_negative = ((prediction == 0) & (target == 0)).sum().item()
    false_positive = ((prediction == 1) & (target == 0)).sum().item()
    false_negative = ((prediction == 0) & (target == 1)).sum().item()
    count = len(target)

    accuracy = (true_positive + true_negative) / count
    return {
        "loss": total_loss / count,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "sensitivity": true_positive / max(true_positive + false_negative, 1),
        "specificity": true_negative / max(true_negative + false_positive, 1),
    }


def train_model(
    model,
    train_data,
    validation_data,
    learning_rate,
    batch_size,
    max_epochs=MAX_EPOCHS,
    early_stopping=True,
    verbose=False,
):
    """Train and return the per-epoch history, restoring the best weights.

    Early stopping monitors validation loss. Validation accuracy, sensitivity
    and specificity are recorded each epoch but never drive training.
    """
    if early_stopping and validation_data is None:
        raise ValueError("early stopping needs a validation set to monitor")

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "validation": []}
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in loader:
            optimiser.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(labels)
            seen += len(labels)

        train_loss = running / seen
        history["train_loss"].append(train_loss)
        metrics = None
        if validation_data is not None:
            metrics = evaluate(model, validation_data)
            history["validation"].append(metrics)

        if verbose:
            message = f"  epoch {epoch:02d} | train loss {train_loss:.4f}"
            if metrics is not None:
                message += (f" | val loss {metrics['loss']:.4f}"
                            f" | val acc {metrics['accuracy']:.4f}")
            print(message)

        if not early_stopping:
            best_epoch = epoch
            continue

        if best_weights is None or metrics["loss"] < best_loss - MIN_IMPROVEMENT:
            best_loss = metrics["loss"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale == PATIENCE:
                if verbose:
                    print(f"  early stopping after epoch {epoch}")
                break

    if early_stopping and best_weights is not None:
        model.load_state_dict(best_weights)

    history["best_epoch"] = best_epoch
    history["epochs_run"] = len(history["train_loss"])
    return history
