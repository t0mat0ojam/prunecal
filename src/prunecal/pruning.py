"""``compress``: structured pruning to a MAC budget, plus the recalibration report.

Pipeline (one call):

1. score the filters of the target convolution (discriminability by
   default, Eq. 1 of the report);
2. find the smallest number of filters to remove that meets the MAC
   budget, ``rho*(R) = min{rho : M(rho) <= M0 / R}`` (Eq. 2), by binary
   search -- or use a fixed ``ratio``;
3. physically remove the lowest-scoring filters (structured pruning via
   torch-pruning, which also updates the affected BatchNorm parameters
   and downstream input channels);
4. run the probe to determine which BatchNorm layers the pruning
   invalidated; these are the layers ``recalibrate`` must repair.

No fine-tuning is performed after pruning (report, Section 1): retraining
would let surviving filters relearn the deleted ones and erase the
difference between selection criteria.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ._utils import LayerRef, resolve_layer
from .criteria import discriminability_scores, magnitude_scores, random_scores
from .macs import count_macs, dominant_conv
from .probe import ProbeResult, probe


class BudgetUnreachableError(RuntimeError):
    """Raised when no pruning ratio of the target layer meets the MAC budget.

    This mirrors the "unreachable" entries of Table 1 in the report: it is
    information, not failure -- even deleting every removable filter of
    this layer does not reduce the model enough. Choose another layer or a
    smaller budget.
    """


@dataclass
class CompressionReport:
    """What ``compress`` did and what must be recalibrated afterwards."""

    layer: str
    criterion: str
    kept: List[int]
    pruned: List[int]
    macs_before: int
    macs_after: int
    budget: Optional[float]
    ratio: float
    recalibrate: List[str]
    """Qualified names of the BatchNorm layers whose statistics the
    pruning invalidated (the probe set). Pass this to
    ``prunecal.recalibrate(model, trials, layers=report.recalibrate)``."""
    probe: ProbeResult = field(repr=False, default=None)
    scores: Dict[int, float] = field(repr=False, default_factory=dict)

    @property
    def achieved_reduction(self) -> float:
        return self.macs_before / self.macs_after

    def summary(self) -> str:
        return (
            f"pruned {len(self.pruned)}/{len(self.pruned) + len(self.kept)} "
            f"filters of {self.layer!r} by {self.criterion} "
            f"(ratio {self.ratio:.3f});  MACs {self.macs_before:,} -> "
            f"{self.macs_after:,}  ({self.achieved_reduction:.2f}x);  "
            f"recalibrate: {self.recalibrate} "
            f"({len(self.recalibrate)}/{self.probe.n_batchnorms} BatchNorm layers)"
        )


def _prune_copy(
    model: nn.Module,
    layer_name: str,
    prune_idx: Sequence[int],
    example_input: torch.Tensor,
) -> nn.Module:
    """Physically remove ``prune_idx`` filters of ``layer_name`` on a copy."""
    from ._graph import resolve

    resolution = resolve(model, layer_name, prune_idx, example_input)
    return resolution.prune_copy(model, example_input)


def _macs_if_pruned(
    model: nn.Module, layer_name: str, k: int, example_input: torch.Tensor
) -> int:
    """Total MACs after removing ``k`` filters (which ones does not matter)."""
    trial = _prune_copy(model, layer_name, list(range(k)), example_input)
    total, _ = count_macs(trial, example_input)
    return total


def compress(
    model: nn.Module,
    data: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    layer: Optional[LayerRef] = None,
    budget: Optional[float] = None,
    ratio: Optional[float] = None,
    criterion: str = "discriminability",
    example_input: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Compress ``model`` and report which layers to recalibrate.

    Parameters
    ----------
    model:
        Trained model. It is never modified; a pruned copy is returned.
    data, labels:
        Training trials ``[N, ...]`` and class labels ``[N]``. Required for
        the default ``criterion="discriminability"``; ``data`` also
        provides the example input if ``example_input`` is not given.
    layer:
        Pruning-target convolution (qualified name or module). If omitted,
        the convolution with the largest MAC share is chosen with a
        warning -- the report (Section 3) shows this default is not always
        accuracy-optimal, so prefer being explicit.
    budget:
        Target MAC reduction factor ``R`` (e.g. ``2.0`` halves the MACs).
        The smallest sufficient number of filters is found by binary
        search; raises ``BudgetUnreachableError`` if the layer cannot
        deliver the budget. Exactly one of ``budget`` / ``ratio`` must be
        given.
    ratio:
        Alternatively, prune a fixed fraction of the layer's filters
        (e.g. ``0.875`` as in the report's primary endpoint).
    criterion:
        ``"discriminability"`` (default; needs ``data`` and ``labels``),
        ``"magnitude"``, or ``"random"`` (baselines from Section 2).

    Returns
    -------
    (pruned_model, report):
        ``report.recalibrate`` names the BatchNorm layers whose running
        statistics the pruning invalidated. After deployment-side data is
        available, call
        ``recalibrate(pruned_model, unlabeled_trials, layers=report.recalibrate)``.
    """
    if (budget is None) == (ratio is None):
        raise ValueError("Give exactly one of `budget` or `ratio`.")
    if example_input is None:
        if data is None:
            raise ValueError("Provide `example_input`, or `data` to take it from.")
        example_input = data[:1]

    if layer is None:
        layer_name = dominant_conv(model, example_input)
        warnings.warn(
            f"No pruning target given; using the MAC-dominant convolution "
            f"{layer_name!r}. The largest layer is not always the best "
            f"target for accuracy (report, Section 3) -- consider choosing "
            f"explicitly.",
            stacklevel=2,
        )
    else:
        layer_name, _ = resolve_layer(model, layer)
    layer_module = dict(model.named_modules())[layer_name]
    n_filters = layer_module.weight.shape[0]

    if criterion == "discriminability":
        if data is None or labels is None:
            raise ValueError("criterion='discriminability' needs `data` and `labels`.")
        scores = discriminability_scores(model, layer_name, data, labels)
    elif criterion == "magnitude":
        scores = magnitude_scores(model, layer_name)
    elif criterion == "random":
        scores = random_scores(model, layer_name, generator=generator)
    else:
        raise ValueError(f"Unknown criterion {criterion!r}.")

    macs_before, _ = count_macs(model, example_input)
    if ratio is not None:
        if not 0.0 < ratio < 1.0:
            raise ValueError("`ratio` must be strictly between 0 and 1.")
        k = min(max(int(round(n_filters * ratio)), 1), n_filters - 1)
    else:
        if budget <= 1.0:
            raise ValueError("`budget` is a reduction factor; it must exceed 1.")
        target = macs_before / budget
        if _macs_if_pruned(model, layer_name, n_filters - 1, example_input) > target:
            raise BudgetUnreachableError(
                f"Layer {layer_name!r} cannot reach a {budget:.2f}x MAC "
                f"reduction even with all removable filters deleted "
                f"(report, Table 1: 'unreachable'). Choose another layer "
                f"or a smaller budget."
            )
        lo, hi = 1, n_filters - 1  # invariant: macs(hi) <= target
        while lo < hi:
            mid = (lo + hi) // 2
            if _macs_if_pruned(model, layer_name, mid, example_input) <= target:
                hi = mid
            else:
                lo = mid + 1
        k = lo

    order = torch.argsort(scores)  # ascending: prune the lowest scores
    pruned_idx = sorted(int(i) for i in order[:k])
    kept_idx = sorted(int(i) for i in order[k:])

    pruned_model = _prune_copy(model, layer_name, pruned_idx, example_input)
    macs_after, _ = count_macs(pruned_model, example_input)

    probe_result = probe(model, layer_name, example_input)
    report = CompressionReport(
        layer=layer_name,
        criterion=criterion,
        kept=kept_idx,
        pruned=pruned_idx,
        macs_before=macs_before,
        macs_after=macs_after,
        budget=budget,
        ratio=k / n_filters,
        recalibrate=list(probe_result.reached),
        probe=probe_result,
        scores={i: float(scores[i]) for i in range(n_filters)},
    )
    return pruned_model, report
