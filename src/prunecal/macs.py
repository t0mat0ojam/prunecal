"""Multiply-accumulate (MAC) counting.

Counts one MAC per (multiply, add) pair in convolution and linear layers,
measured by registering hooks and running one example input through the
model, so the number depends only on the architecture and the input shape
(report, Section 3). Use a batch of one trial for per-trial numbers.
These are theoretical operation counts, distinct from on-device latency.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


@torch.no_grad()
def count_macs(
    model: nn.Module, example_input: torch.Tensor
) -> Tuple[int, Dict[str, int]]:
    """Return ``(total_macs, per_layer_macs)`` for one forward pass."""
    per_layer: Dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            if isinstance(module, _CONVS):
                kernel_ops = (
                    module.in_channels // module.groups
                ) * math.prod(module.kernel_size)
                macs = output.numel() * kernel_ops
            elif isinstance(module, nn.Linear):
                macs = output.numel() // output.shape[-1]
                macs = macs * module.out_features * module.in_features
            else:
                return
            per_layer[name] = per_layer.get(name, 0) + int(macs)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, _CONVS + (nn.Linear,)):
            handles.append(module.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        model(example_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return sum(per_layer.values()), per_layer


def dominant_conv(model: nn.Module, example_input: torch.Tensor) -> str:
    """Name of the convolution with the largest MAC share.

    Note: the report (Section 3) shows the largest layer is *not* always
    the best pruning target for accuracy (e.g. ShallowConvNet's spatial
    convolution). Prefer passing the target layer explicitly.
    """
    _, per_layer = count_macs(model, example_input)
    convs = {
        n: v
        for n, v in per_layer.items()
        if isinstance(dict(model.named_modules())[n], _CONVS)
    }
    if not convs:
        raise ValueError("Model contains no convolution layers.")
    return max(convs, key=convs.get)
