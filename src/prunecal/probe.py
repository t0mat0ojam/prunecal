"""The probe: which normalization layers does pruning actually reach?

Procedure (report, Section 5, Fig. 5):

(i)   run a fixed input through an unmodified copy of the model and record
      the tensor entering every BatchNorm layer;
(ii)  zero out a single filter of the pruning-target layer;
(iii) record the same tensors again;
(iv)  compare exactly, layer by layer. A layer whose values changed holds
      statistics that pruning invalidates ("reached"); a layer whose
      values did not change is untouched by pruning.

Channels that change *trivially* -- i.e. the channels that structured
pruning would physically delete anyway -- are excluded from the
comparison; including them would flag every layer as reached. The set of
deleted channels per layer comes from the same dependency resolution used
for the surgery itself (``prunecal._graph``): torch-pruning's dependency
graph where its result validates, and an exact propagation along the
module chain otherwise -- so the exclusion is exact for the grouped and
depthwise convolutions of EEGNet-style models.

The answer depends only on the network wiring, not on the weights: a
randomly initialized model gives the same result, no data or training is
required, and the whole check runs in seconds. This is what makes it
cheap to run on any published BatchNorm CNN (EMG, ECG, vision, ...).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ._utils import LayerRef, batchnorms, resolve_layer


@dataclass
class ProbeResult:
    """Outcome of one probe run."""

    layer: str
    """Qualified name of the pruning-target layer that was probed."""
    filter_index: int
    """Index of the filter that was zeroed."""
    reached: List[str]
    """BatchNorm layers whose recorded statistics pruning invalidates."""
    untouched: List[str]
    """BatchNorm layers that pruning does not reach."""
    max_change: Dict[str, float] = field(default_factory=dict)
    """Largest absolute change of the pre-BN tensor on surviving channels."""

    @property
    def n_batchnorms(self) -> int:
        return len(self.reached) + len(self.untouched)

    def summary(self) -> str:
        lines = [
            f"probe of {self.layer!r} (filter {self.filter_index}): "
            f"{len(self.reached)} / {self.n_batchnorms} BatchNorm layers reached"
        ]
        for name in self.reached:
            lines.append(f"  reached   {name}  (max change {self.max_change[name]:.3e})")
        for name in self.untouched:
            lines.append(f"  untouched {name}")
        return "\n".join(lines)


@torch.no_grad()
def _pre_bn_inputs(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Tensor entering each BatchNorm layer for input ``x``."""
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in batchnorms(model):
        def hook(mod, inputs, _name=name):
            captured[_name] = inputs[0].detach().clone()
        handles.append(module.register_forward_pre_hook(hook))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def _deleted_channels_per_bn(
    model: nn.Module, layer_name: str, filter_index: int, example_input: torch.Tensor
) -> Dict[str, Set[int]]:
    """Channels each BN would lose if ``filter_index`` were physically pruned.

    Computed via the pruning dependency resolver (torch-pruning where its
    result validates, an exact sequential-chain propagation otherwise);
    these are the "trivially changing" channels excluded from the probe
    comparison.
    """
    from ._graph import resolve

    return resolve(model, layer_name, [filter_index], example_input).bn_deletions


def probe(
    model: nn.Module,
    layer: LayerRef,
    example_input: torch.Tensor,
    filter_index: int = 0,
    atol: float = 0.0,
) -> ProbeResult:
    """Determine which BatchNorm layers pruning ``layer`` reaches.

    Parameters
    ----------
    model:
        Any ``nn.Module`` containing BatchNorm layers. The model is
        deep-copied; the original is never modified. Weights can be
        random -- the answer is determined by the wiring.
    layer:
        The pruning-target convolution, by qualified name or module.
    example_input:
        One input of the correct shape (e.g. ``torch.randn(1, 1, C, T)``).
    filter_index:
        Which filter to zero. The result is wiring-determined and should
        not depend on this; vary it to verify. One caveat: in a ReLU
        network at *random* initialization, a filter can be functionally
        dead (its bias dominates at every input amplitude), in which case
        zeroing it truly changes nothing. The probe defends itself by
        evaluating a battery of scaled inputs, but if every layer comes
        back untouched, try another ``filter_index`` or trained weights.
    atol:
        Comparison tolerance. The default ``0.0`` is the exact comparison
        used in the report: untouched layers execute bit-identical
        operations, so any nonzero difference means the layer is reached.
    """
    work = copy.deepcopy(model)
    work.eval()
    layer_name, layer_module = resolve_layer(work, layer if isinstance(layer, str) else resolve_layer(model, layer)[0])
    n_filters = layer_module.weight.shape[0]
    if not 0 <= filter_index < n_filters:
        raise IndexError(
            f"filter_index {filter_index} out of range for {layer_name!r} "
            f"with {n_filters} filters."
        )

    excluded = _deleted_channels_per_bn(
        work, layer_name, filter_index, example_input
    )

    # The answer must depend on the wiring, not on the weights -- but with
    # a single input, an unlucky initialization can gate the perturbation
    # to exactly zero (e.g. a ReLU channel that happens to be negative
    # everywhere), hiding a reached layer. Probing a small battery of
    # scaled inputs (+-x, +-8x) makes that coincidence require the channel
    # to be dead at every amplitude and sign simultaneously, which no
    # standard initialization produces. Untouched layers stay bit-identical
    # on every battery element, so this adds no false positives.
    battery = [example_input]
    if example_input.is_floating_point():
        battery += [-example_input, 8.0 * example_input, -8.0 * example_input]

    baselines = [_pre_bn_inputs(work, x) for x in battery]
    with torch.no_grad():
        layer_module.weight[filter_index].zero_()
        if layer_module.bias is not None:
            layer_module.bias[filter_index].zero_()
    perturbeds = [_pre_bn_inputs(work, x) for x in battery]

    reached: List[str] = []
    untouched: List[str] = []
    max_change: Dict[str, float] = {}
    for name, _ in batchnorms(work):
        keep = [
            c
            for c in range(baselines[0][name].shape[1])
            if c not in excluded.get(name, set())
        ]
        if not keep:
            # Every channel of this layer is deleted by the pruning itself.
            untouched.append(name)
            max_change[name] = 0.0
            continue
        diff = max(
            (before[name][:, keep] - after[name][:, keep]).abs().max().item()
            for before, after in zip(baselines, perturbeds)
        )
        max_change[name] = diff
        if diff > atol:
            reached.append(name)
        else:
            untouched.append(name)
    return ProbeResult(
        layer=layer_name,
        filter_index=filter_index,
        reached=reached,
        untouched=untouched,
        max_change=max_change,
    )
