"""Reference models.

``EEGNet`` is a self-contained implementation of the EEGNet-v4
architecture (Lawhern et al., 2018) so the pipeline and its tests run
without external EEG dependencies. It follows the standard structure
(temporal conv -> BN -> depthwise spatial conv -> BN -> separable conv ->
BN -> classifier); if you trained with another implementation (e.g.
braindecode's ``EEGNetv4``), use *your* trained model with the pipeline
and pass the appropriate layer name -- the pipeline works on any
``nn.Module``.

The three ``Control*`` models are the synthetic controls of Table 2 of
the report: architectures whose probe outcome is known from their wiring
(depthwise-only: 1 of 3 BN layers reached; dense: 2 of 3; skip
connection: 2 of 4). They exist to validate the probe and to illustrate
why the "recalibrate one layer" result holds for EEGNet but not for
most modern CNNs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet-v4-style compact CNN for EEG classification.

    Input shape: ``[batch, 1, n_channels, n_samples]``.

    The report's configuration is ``F1=32, D=2, F2=64,
    kernel_length=64, separable_length=16, pool=(4, 8), dropout=0.25``.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        separable_length: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn_temporal = nn.BatchNorm2d(F1)
        self.conv_spatial = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn_spatial = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.conv_separable_depth = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, separable_length),
            groups=F1 * D,
            padding=(0, separable_length // 2),
            bias=False,
        )
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn_separable = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            probe_in = torch.zeros(1, 1, n_channels, n_samples)
            n_features = self._features(probe_in).shape[1]
        self.classifier = nn.Linear(n_features, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_temporal(self.conv_temporal(x))
        x = self.bn_spatial(self.conv_spatial(x))
        x = self.drop1(self.pool1(torch.nn.functional.elu(x)))
        x = self.conv_separable_point(self.conv_separable_depth(x))
        x = self.bn_separable(x)
        x = self.drop2(self.pool2(torch.nn.functional.elu(x)))
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x))


class ControlDepthwise(nn.Module):
    """Depthwise-only control (Table 2: 3 BN layers, 1 reached).

    Channel-mixing happens only in the final pointwise convolution, so
    pruning ``conv1`` reaches only ``bn3`` -- the EEGNet situation.
    """

    def __init__(self, in_channels: int = 3, width: int = 8, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, width, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv_dw = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv_pw = nn.Conv1d(width, width, 1)
        self.bn3 = nn.BatchNorm1d(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.elu(self.bn1(self.conv1(x)))
        x = nn.functional.elu(self.bn2(self.conv_dw(x)))
        x = nn.functional.elu(self.bn3(self.conv_pw(x)))
        return self.head(x.mean(dim=-1))


class ControlDense(nn.Module):
    """Dense-convolution control (Table 2: 3 BN layers, 2 reached).

    Every convolution mixes all input channels, so pruning ``conv1``
    reaches every downstream BN (``bn2``, ``bn3``) -- only ``bn1`` is
    untouched.
    """

    def __init__(self, in_channels: int = 3, width: int = 8, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, width, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, width, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.elu(self.bn1(self.conv1(x)))
        x = nn.functional.elu(self.bn2(self.conv2(x)))
        x = nn.functional.elu(self.bn3(self.conv3(x)))
        return self.head(x.mean(dim=-1))


class ControlSkip(nn.Module):
    """Skip-connection control (Table 2: 4 BN layers, 2 reached).

    Same depthwise-separable block as ``ControlDepthwise``, but a shortcut
    bypasses the mixing. The shortcut carries the perturbation past the
    depthwise isolation, so ``bn4`` (after the addition) is reached in
    addition to ``bn3`` -- the mechanism that makes MobileNetV2-style
    networks require near-full recalibration despite their depthwise
    structure.
    """

    def __init__(self, in_channels: int = 3, width: int = 8, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, width, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv_dw = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv_pw = nn.Conv1d(width, width, 1)
        self.bn3 = nn.BatchNorm1d(width)
        self.bn4 = nn.BatchNorm1d(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = nn.functional.elu(self.bn1(self.conv1(x)))
        x = nn.functional.elu(self.bn2(self.conv_dw(shortcut)))
        x = self.bn3(self.conv_pw(x))
        x = self.bn4(nn.functional.elu(x + shortcut))
        return self.head(x.mean(dim=-1))
