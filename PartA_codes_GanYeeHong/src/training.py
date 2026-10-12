"""Reusable training and evaluation loop shared by the Part A questions.

Generalises ``baseline.train_model`` in three ways that the assignment needs:

* records the per-epoch training and evaluation RMSE so learning curves can be
  plotted afterwards,
* exposes an ``on_epoch_end`` callback so Ray Tune can report each epoch, and
* allows early stopping to be switched off for the combined train+validation
  retrain, where no held-out split is left to monitor.

Optimiser, epoch budget and early-stopping rule follow the assignment brief:
Adam, at most 20 epochs, patience 4, minimum improvement 1e-4 (see
``config.Config``).
"""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config


@torch.no_grad()
def predict(model, dataset, batch_size):
    """Return (predicted price, actual price) over a dataset, in dollars."""
    model.eval()
    predictions = []
    targets = []
    for categorical, continuous, target in DataLoader(dataset, batch_size=batch_size):
        predictions.append(model.predict_price(categorical, continuous))
        targets.append(target)
    return torch.cat(predictions), torch.cat(targets)


def evaluate(model, dataset, batch_size):
    """Return (RMSE, R^2) in dollars.

    R^2 is computed against the mean of the dataset being evaluated, which is
    the convention used by ``sklearn.metrics.r2_score``: the "predict the
    average" reference must be measured on the same rows that are being scored.
    """
    prediction, target = predict(model, dataset, batch_size)
    squared_error = (prediction - target).square().sum()
    total_variation = (target - target.mean()).square().sum()
    rmse = (squared_error / len(target)).sqrt().item()
    r_squared = (1.0 - squared_error / total_variation).item()
    return rmse, r_squared


def calculate_rmse(model, dataset, batch_size):
    """RMSE in dollars."""
    return evaluate(model, dataset, batch_size)[0]


def train_model(
    model,
    train_data,
    config: Config,
    evaluation_data=None,
    max_epochs=None,
    early_stopping=True,
    on_epoch_end=None,
    verbose=True,
    record_r2=False,
):
    """Train ``model`` with Adam and return the per-epoch history.

    With ``early_stopping=True`` (model selection on the 2021 validation year)
    the weights from the epoch with the lowest evaluation RMSE are restored at
    the end, and training stops once ``config.early_stopping_patience`` epochs
    pass without an improvement of at least ``config.early_stopping_threshold``.

    With ``early_stopping=False`` the loop runs for exactly ``max_epochs`` and
    keeps the final weights. This is the combined train+validation retrain:
    validation has been folded into training, so there is nothing left to
    monitor, and the epoch budget is instead fixed to the value selected during
    model selection. ``evaluation_data`` may still be passed to record a curve,
    but it never influences training or weight selection.

    Returns a dict with ``train_rmse``, ``evaluation_rmse``, ``best_epoch`` and
    ``best_evaluation_rmse``.
    """
    epoch_budget = config.max_epochs if max_epochs is None else max_epochs
    if early_stopping and evaluation_data is None:
        raise ValueError("early stopping needs evaluation_data to monitor")

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {"train_rmse": [], "evaluation_rmse": [],
               "train_fit_rmse": [], "train_r2": [], "evaluation_r2": []}
    best_rmse = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epoch_budget + 1):
        model.train()
        squared_error = 0.0
        row_count = 0

        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()
            standard_prediction = model(categorical, continuous)
            standard_target = (target - model.target_mean) / model.target_std
            loss = loss_function(standard_prediction, standard_target)
            loss.backward()
            optimiser.step()

            prediction = (
                standard_prediction.detach() * model.target_std + model.target_mean
            )
            squared_error += (prediction - target).square().sum().item()
            row_count += len(target)

        train_rmse = (squared_error / row_count) ** 0.5
        history["train_rmse"].append(train_rmse)

        evaluation_rmse = None
        if evaluation_data is not None:
            if record_r2:
                evaluation_rmse, evaluation_r2 = evaluate(
                    model, evaluation_data, config.batch_size
                )
                history["evaluation_r2"].append(evaluation_r2)
            else:
                evaluation_rmse = calculate_rmse(
                    model, evaluation_data, config.batch_size
                )
            history["evaluation_rmse"].append(evaluation_rmse)

        if record_r2:
            # Train metrics measured at epoch end, so they are comparable with
            # the validation metrics rather than averaged over a changing model.
            # Iterating a DataLoader draws from the global RNG, so the state is
            # saved and restored: recording these metrics must not alter the
            # minibatch order and therefore the training trajectory.
            rng_state = torch.get_rng_state()
            fit_rmse, fit_r2 = evaluate(model, train_data, config.batch_size)
            torch.set_rng_state(rng_state)
            history["train_fit_rmse"].append(fit_rmse)
            history["train_r2"].append(fit_r2)

        if verbose:
            message = f"epoch {epoch:02d} | train RMSE {train_rmse:,.0f}"
            if evaluation_rmse is not None:
                message += f" | evaluation RMSE {evaluation_rmse:,.0f}"
            print(message)

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_rmse, evaluation_rmse)

        if not early_stopping:
            best_epoch = epoch
            continue

        if (
            best_weights is None
            or evaluation_rmse < best_rmse - config.early_stopping_threshold
        ):
            best_rmse = evaluation_rmse
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                if verbose:
                    print(f"early stopping after epoch {epoch}")
                break

    if early_stopping:
        model.load_state_dict(best_weights)

    history["best_epoch"] = best_epoch
    history["best_evaluation_rmse"] = best_rmse if early_stopping else None
    return history


def residual_diagnostics(model, dataset, batch_size):
    """Decompose the error on a dataset into a bias and a spread term.

    RMSE alone cannot distinguish a model that is systematically off from one
    that is merely noisy. Writing the mean squared error as

        MSE = bias^2 + variance(residual)

    separates the two: ``bias`` is the mean residual (negative means the model
    underpredicts) and ``bias_share`` is the fraction of the MSE it explains. A
    large ``bias_share`` indicates distribution shift rather than overfitting.
    """
    prediction, target = predict(model, dataset, batch_size)
    residual = prediction - target
    bias = residual.mean()
    mean_squared_error = residual.square().mean()
    return {
        "mean_actual": target.mean().item(),
        "mean_predicted": prediction.mean().item(),
        "bias": bias.item(),
        "residual_sd": residual.std().item(),
        "bias_share_of_mse": (bias.square() / mean_squared_error).item(),
    }


def parameter_breakdown(model):
    """Return ([(readable name, shape, count)], total) for all trainable tensors."""
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rows.append((name, tuple(parameter.shape), parameter.numel()))
    return rows, sum(count for _, _, count in rows)
