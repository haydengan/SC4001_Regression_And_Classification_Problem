"""The three CNN architectures of Figure B.

Every convolution uses SAME padding, BatchNorm and no bias; fully connected
layers keep their bias. Layers marked /2 in the figure use stride 2.

In B3 the second convolution of each residual block is followed by BatchNorm
only: the ReLU comes after the residual addition, which is what the figure
caption means by "ReLU ... including after residual addition".
"""

import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class AlexNetLike(nn.Module):
    """B1. Dropout 0.5 is applied only in the hidden fully connected layers."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16, 5),
            nn.MaxPool2d(2),
            conv_block(16, 32, 5),
            nn.MaxPool2d(2),
            conv_block(32, 64, 3),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class VGGLike(nn.Module):
    """B2."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 8, 3),
            conv_block(8, 8, 3),
            nn.MaxPool2d(2),
            conv_block(8, 16, 3),
            conv_block(16, 16, 3),
            nn.MaxPool2d(2),
            conv_block(16, 32, 3),
            conv_block(32, 32, 3),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection; ReLU after the addition.

    When the block changes resolution or width the skip path uses a 1x1
    convolution with the same stride, as drawn in Figure B3.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.first = conv_block(in_channels, out_channels, 3, stride=stride)
        self.second = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.second(self.first(x)) + self.skip(x))


class ResNetLike(nn.Module):
    """B3."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16, 3),
            nn.MaxPool2d(2),
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


ARCHITECTURES = {
    "B1 AlexNet-like": AlexNetLike,
    "B2 VGG-like": VGGLike,
    "B3 ResNet-like": ResNetLike,
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape=(1, 1, 64, 64)):
    """Forward-pass FLOPs of the convolution and linear layers.

    One multiply-accumulate is counted as two floating-point operations.
    """
    multiply_accumulates = 0
    handles = []

    def convolution(module, _inputs, output):
        nonlocal multiply_accumulates
        kernel = module.kernel_size[0] * module.kernel_size[1]
        per_position = (module.in_channels // module.groups) * kernel
        multiply_accumulates += (
            per_position * module.out_channels * output.shape[2] * output.shape[3]
        )

    def linear(module, _inputs, _output):
        nonlocal multiply_accumulates
        multiply_accumulates += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(convolution))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return 2 * multiply_accumulates
