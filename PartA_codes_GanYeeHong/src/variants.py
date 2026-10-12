"""Model variants for Questions A2 and A3.

Covers the A1 architecture, the two A2 ablations (no categorical embeddings,
Sigmoid instead of ReLU) and the A3 wide-and-deep model, which adds a linear
branch over the same concatenated features:

    output = MLP output + lambda * linear output
"""

import torch
from torch import nn

from config import Config

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}


class PriceModelVariant(nn.Module):
    def __init__(
        self,
        cardinalities,
        target_mean: float,
        target_std: float,
        config: Config,
        use_categorical: bool = True,
        activation: str = "relu",
        wide_lambda: float | None = None,
    ) -> None:
        super().__init__()
        self.use_categorical = use_categorical
        self.wide_lambda = wide_lambda

        if use_categorical:
            self.embeddings = nn.ModuleList(
                nn.Embedding(cardinality, width)
                for cardinality, width in zip(cardinalities, config.embedding_dims)
            )
            input_width = sum(config.embedding_dims) + len(config.continuous_features)
        else:
            self.embeddings = nn.ModuleList()
            input_width = len(config.continuous_features)

        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            ACTIVATIONS[activation](),
            nn.Linear(config.hidden_width, 1),
        )
        self.wide = nn.Linear(input_width, 1) if wide_lambda is not None else None

        self.target_mean = target_mean
        self.target_std = target_std

    def features(self, categorical: torch.Tensor, continuous: torch.Tensor):
        if not self.use_categorical:
            return continuous
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(embedded + [continuous], dim=1)

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor):
        features = self.features(categorical, continuous)
        output = self.mlp(features).squeeze(1)
        if self.wide is not None:
            output = output + self.wide_lambda * self.wide(features).squeeze(1)
        return output

    def predict_price(self, categorical: torch.Tensor, continuous: torch.Tensor):
        return self(categorical, continuous) * self.target_std + self.target_mean


def build_variant(data, config: Config, **kwargs) -> PriceModelVariant:
    return PriceModelVariant(
        data["cardinalities"], data["target_mean"], data["target_std"], config, **kwargs
    )
