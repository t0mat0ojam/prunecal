"""Filter-selection criteria for structured pruning.

The main criterion is *discriminability* (Section 2 of the report): each
filter's response to a trial is summarized as log band power, and filters
are scored by a one-way ANOVA F-statistic (between-class variance divided
by within-class variance, Eq. 1). High scores mean the filter's output
separates the classes -- for motor imagery, these are the filters that
captured event-related desynchronization.

``magnitude_scores`` and ``random_scores`` are included as the baselines
the report compares against (both were indistinguishable from random /
uninformative at the primary endpoint).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ._utils import LayerRef, resolve_layer


@torch.no_grad()
def _layer_activations(
    model: nn.Module,
    layer: nn.Module,
    data: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """Collect the output of ``layer`` for every trial in ``data``."""
    captured = []
    handle = layer.register_forward_hook(
        lambda mod, inp, out: captured.append(out.detach())
    )
    was_training = model.training
    model.eval()
    try:
        for start in range(0, data.shape[0], batch_size):
            model(data[start : start + batch_size])
    finally:
        handle.remove()
        model.train(was_training)
    return torch.cat(captured, dim=0)


def log_power(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Summarize activations ``[N, F, ...]`` as per-trial log power ``[N, F]``.

    Band-pass-filtered signals have near-zero temporal mean, so the mean of
    the squared activation is the band power (report, Section 2).
    """
    reduce_dims = tuple(range(2, activations.dim()))
    power = activations.pow(2)
    if reduce_dims:
        power = power.mean(dim=reduce_dims)
    return torch.log(power + eps)


def discriminability_scores(
    model: nn.Module,
    layer: LayerRef,
    data: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. 1 of the report: one-way ANOVA F-statistic per filter.

    ``data`` are training trials ``[N, ...]`` and ``labels`` their class
    labels ``[N]``. Returns one score per output filter of ``layer``;
    higher means more class-discriminative (keep), lower means prune first.
    """
    _, layer_module = resolve_layer(model, layer)
    phi = log_power(_layer_activations(model, layer_module, data, batch_size))
    labels = torch.as_tensor(labels).reshape(-1)
    if labels.shape[0] != phi.shape[0]:
        raise ValueError(
            f"Got {phi.shape[0]} trials but {labels.shape[0]} labels."
        )
    classes = labels.unique()
    n_classes, n_trials = len(classes), phi.shape[0]
    if n_classes < 2:
        raise ValueError("Discriminability needs at least two classes.")
    grand_mean = phi.mean(dim=0)
    between = torch.zeros_like(grand_mean)
    within = torch.zeros_like(grand_mean)
    for k in classes:
        mask = labels == k
        n_k = int(mask.sum())
        class_mean = phi[mask].mean(dim=0)
        between += n_k * (class_mean - grand_mean) ** 2
        within += ((phi[mask] - class_mean) ** 2).sum(dim=0)
    between = between / (n_classes - 1)
    within = within / (n_trials - n_classes)
    return between / (within + eps)


def magnitude_scores(model: nn.Module, layer: LayerRef) -> torch.Tensor:
    """L1 weight magnitude per filter (the standard baseline)."""
    _, layer_module = resolve_layer(model, layer)
    weight = layer_module.weight.detach()
    return weight.abs().sum(dim=tuple(range(1, weight.dim())))


def random_scores(
    model: nn.Module,
    layer: LayerRef,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Uniform random scores (the lower-bound control)."""
    _, layer_module = resolve_layer(model, layer)
    n_filters = layer_module.weight.shape[0]
    return torch.rand(n_filters, generator=generator)
