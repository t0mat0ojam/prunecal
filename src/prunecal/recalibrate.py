"""Label-free recalibration of BatchNorm statistics, and the Delta metric.

``recalibrate`` re-estimates the running mean and standard deviation of
selected BatchNorm layers from one pass over unlabeled trials (16 trials
in the report). Restricted to ``report.recalibrate`` this is the probe-set
recalibration of Section 5; with ``layers=None`` it is full AdaBN. No
labels, no backpropagation, no input-side transform -- which is why it
fits on a microcontroller.

``delta`` measures, per BatchNorm layer, how wrong the stored statistics
are for given data (Eq. 3 of the report):

    Delta = ((mu_t - mu_s) / sigma_s)^2 + (sigma_t / sigma_s - 1)^2

averaged over channels, where (mu_s, sigma_s) are the stored running
statistics and (mu_t, sigma_t) are the statistics of the data actually
entering the layer. A correct record gives Delta = 0; recalibration
drives it there by construction.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn as nn

from ._utils import LayerRef, select_batchnorms
from .probe import _pre_bn_inputs


@torch.no_grad()
def recalibrate(
    model: nn.Module,
    data: torch.Tensor,
    layers: Union[None, Iterable[LayerRef]] = None,
    batch_size: Optional[int] = None,
) -> nn.Module:
    """Re-estimate BatchNorm statistics of ``layers`` from ``data``.

    Parameters
    ----------
    model:
        Model to recalibrate. Modified in place and also returned.
    data:
        Unlabeled trials ``[N, ...]`` from the target session/domain.
        Labels are never used. The report uses 16 trials.
    layers:
        BatchNorm layers to recalibrate, e.g. ``report.recalibrate`` from
        ``compress``. ``None`` recalibrates every BatchNorm layer
        (= full AdaBN).
    batch_size:
        Process the trials in chunks of this size; by default all trials
        form a single batch. Statistics are accumulated as a cumulative
        average either way, so the result is a one-pass estimate.
    """
    selected = select_batchnorms(model, layers)
    saved = [(bn, bn.momentum, bn.training) for _, bn in selected]

    was_training = model.training
    model.eval()
    for _, bn in selected:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average across batches
        bn.train()
    try:
        step = batch_size or data.shape[0]
        for start in range(0, data.shape[0], step):
            model(data[start : start + step])
    finally:
        for bn, momentum, training in saved:
            bn.momentum = momentum
            bn.train(training)
        model.train(was_training)
    return model


@torch.no_grad()
def delta(
    model: nn.Module,
    data: torch.Tensor,
    layers: Union[None, Iterable[LayerRef]] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Eq. 3 of the report, per BatchNorm layer, averaged over channels."""
    selected = dict(select_batchnorms(model, layers))
    was_training = model.training
    model.eval()
    try:
        pre_inputs = _pre_bn_inputs(model, data)
    finally:
        model.train(was_training)

    out: Dict[str, float] = {}
    for name, bn in selected.items():
        x = pre_inputs[name]
        reduce_dims = (0,) + tuple(range(2, x.dim()))
        mu_t = x.mean(dim=reduce_dims)
        var_t = x.var(dim=reduce_dims, unbiased=False)
        mu_s = bn.running_mean
        var_s = bn.running_var
        term_mean = (mu_t - mu_s) ** 2 / (var_s + eps)
        term_std = (torch.sqrt((var_t + eps) / (var_s + eps)) - 1.0) ** 2
        out[name] = float((term_mean + term_std).mean())
    return out
