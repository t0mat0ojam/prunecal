"""Channel-dependency resolution for structured pruning.

Removing output filters of a convolution forces coordinated deletions
downstream: the following BatchNorm loses those channels, grouped and
depthwise convolutions lose whole groups (and therefore output channels),
the first channel-mixing layer (a dense convolution or a linear layer)
loses input channels -- and there the structural propagation stops.

Two engines compute this deletion plan:

* ``torch-pruning``'s dependency graph, which traces arbitrary graphs
  (residual connections included). Its result is *validated* here by
  executing the surgery on a scratch copy and checking structural
  consistency plus a forward pass, because some patterns -- notably
  grouped convolutions with a depth multiplier, as in EEGNet's spatial
  convolution -- are not propagated correctly by it.
* A built-in exact propagation for single-path (sequential) module
  chains, used as the fallback. It handles dense, depthwise, and
  grouped-with-multiplier convolutions, BatchNorm, and linear heads
  (including flattened features).

Both produce the same two artefacts: a way to physically prune a copy of
the model, and the per-BatchNorm sets of deleted channels that the probe
must exclude from its comparison.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_PARAMETRIC = _CONVS + (nn.Linear,) + (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class UnsupportedGraphError(RuntimeError):
    """Neither engine could resolve the pruning dependencies of this model."""


# --------------------------------------------------------------------------
# Engine 1: torch-pruning, with validation
# --------------------------------------------------------------------------

def _structurally_consistent(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, _CONVS):
            if module.out_channels <= 0 or module.in_channels <= 0:
                return False
            if module.weight.shape[0] != module.out_channels:
                return False
            if module.weight.shape[1] * module.groups != module.in_channels:
                return False
            if module.weight.shape[1] <= 0:
                return False
        elif isinstance(module, _BatchNorm):
            if module.num_features <= 0:
                return False
            if module.weight is not None and module.weight.shape[0] != module.num_features:
                return False
    return True


def _tp_prune(model: nn.Module, layer_name: str, idxs: Sequence[int],
              example_input: torch.Tensor) -> Dict[str, Set[int]]:
    """Prune ``model`` in place with torch-pruning; return BN deletions.

    Raises if torch-pruning is unavailable or produced an invalid model.
    """
    import torch_pruning as tp

    layer = dict(model.named_modules())[layer_name]
    names = {m: n for n, m in model.named_modules()}
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
    group = graph.get_pruning_group(
        layer, tp.prune_conv_out_channels, idxs=sorted(int(i) for i in idxs)
    )
    bn_deletions: Dict[str, Set[int]] = {}
    for dep, dep_idxs in group:
        target = dep.target.module
        if isinstance(target, _BatchNorm) and target in names:
            bn_deletions.setdefault(names[target], set()).update(int(i) for i in dep_idxs)
    group.prune()
    if not _structurally_consistent(model):
        raise UnsupportedGraphError("torch-pruning produced an inconsistent model.")
    with torch.no_grad():
        model(example_input)  # must run
    return bn_deletions


# --------------------------------------------------------------------------
# Engine 2: exact propagation along a sequential parametric chain
# --------------------------------------------------------------------------

def _execution_order(model: nn.Module, example_input: torch.Tensor) -> List[Tuple[str, nn.Module]]:
    order: List[Tuple[str, nn.Module]] = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, _PARAMETRIC):
            handles.append(
                module.register_forward_hook(
                    lambda mod, i, o, _name=name: order.append((_name, mod))
                )
            )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    seen = [n for n, _ in order]
    if len(set(seen)) != len(seen):
        raise UnsupportedGraphError("A module is executed more than once (weight sharing).")
    return order


def _sequential_plan(
    model: nn.Module, layer_name: str, idxs: Sequence[int], example_input: torch.Tensor
) -> "List[Tuple[str, str, dict]]":
    """Deletion plan [(name, kind, spec)] assuming single-path dataflow.

    The plan's validity is checked afterwards by applying it to a scratch
    copy and running a forward pass; branching graphs fail that check and
    raise ``UnsupportedGraphError``.
    """
    order = _execution_order(model, example_input)
    names_in_order = [n for n, _ in order]
    if layer_name not in names_in_order:
        raise UnsupportedGraphError(f"{layer_name!r} was not executed by the model.")
    start = names_in_order.index(layer_name)
    target = order[start][1]
    if not isinstance(target, _CONVS) or target.groups != 1:
        raise UnsupportedGraphError(
            "The sequential engine only supports pruning dense (groups=1) convolutions."
        )
    dead: Set[int] = set(int(i) for i in idxs)
    if not dead or max(dead) >= target.out_channels:
        raise IndexError("Pruning indices out of range.")

    plan: List[Tuple[str, str, dict]] = [(layer_name, "conv_out", {"idxs": sorted(dead)})]
    for name, module in order[start + 1 :]:
        if not dead:
            break
        if isinstance(module, _BatchNorm):
            if module.num_features <= max(dead):
                raise UnsupportedGraphError(f"Channel bookkeeping broke at {name!r}.")
            plan.append((name, "bn", {"idxs": sorted(dead)}))
        elif isinstance(module, _CONVS):
            if module.in_channels <= max(dead):
                raise UnsupportedGraphError(f"Channel bookkeeping broke at {name!r}.")
            if module.groups == 1:
                plan.append((name, "conv_in", {"idxs": sorted(dead)}))
                dead = set()
            else:
                group_size = module.in_channels // module.groups
                multiplier = module.out_channels // module.groups
                dead_groups = sorted({c // group_size for c in dead})
                covered = {g * group_size + k for g in dead_groups for k in range(group_size)}
                if covered != dead:
                    raise UnsupportedGraphError(
                        f"Pruning removes partial groups of {name!r}; unsupported."
                    )
                dead_out = sorted(
                    g * multiplier + k for g in dead_groups for k in range(multiplier)
                )
                plan.append(
                    (
                        name,
                        "conv_grouped",
                        {
                            "in_idxs": sorted(dead),
                            "out_idxs": dead_out,
                            "dead_groups": len(dead_groups),
                        },
                    )
                )
                dead = set(dead_out)
        elif isinstance(module, nn.Linear):
            # The tensor was (possibly) flattened; channels map to
            # contiguous feature blocks of equal size. Recover the channel
            # count of the incoming tensor from the previous plan step.
            prev_channels = _channels_before_linear(plan, model)
            if prev_channels is None or module.in_features % prev_channels:
                raise UnsupportedGraphError(
                    f"Cannot map channels to features of {name!r}."
                )
            per_channel = module.in_features // prev_channels
            feat = sorted(
                c * per_channel + k for c in dead for k in range(per_channel)
            )
            plan.append((name, "linear_in", {"idxs": feat}))
            dead = set()
        else:  # pragma: no cover - _PARAMETRIC covers the above
            raise UnsupportedGraphError(f"Unhandled module {name!r}.")
    return plan


def _channels_before_linear(plan, model: nn.Module):
    named = dict(model.named_modules())
    for name, kind, _ in reversed(plan):
        module = named[name]
        if kind in ("conv_out",):
            return module.out_channels
        if kind == "conv_grouped":
            return module.out_channels
        if kind == "bn":
            return module.num_features
    return None


def _slice_param(param, keep, dim=0):
    if param is None:
        return None
    return nn.Parameter(
        param.detach().index_select(dim, torch.tensor(keep, dtype=torch.long)).clone(),
        requires_grad=param.requires_grad,
    )


def _apply_plan(model: nn.Module, plan) -> None:
    named = dict(model.named_modules())
    for name, kind, spec in plan:
        module = named[name]
        if kind == "conv_out":
            keep = [i for i in range(module.out_channels) if i not in set(spec["idxs"])]
            module.weight = _slice_param(module.weight, keep, dim=0)
            module.bias = _slice_param(module.bias, keep, dim=0)
            module.out_channels = len(keep)
        elif kind == "conv_in":
            keep = [i for i in range(module.in_channels) if i not in set(spec["idxs"])]
            module.weight = _slice_param(module.weight, keep, dim=1)
            module.in_channels = len(keep)
        elif kind == "conv_grouped":
            keep_out = [i for i in range(module.out_channels) if i not in set(spec["out_idxs"])]
            module.weight = _slice_param(module.weight, keep_out, dim=0)
            module.bias = _slice_param(module.bias, keep_out, dim=0)
            module.out_channels = len(keep_out)
            module.in_channels = module.in_channels - len(spec["in_idxs"])
            module.groups = module.groups - spec["dead_groups"]
        elif kind == "bn":
            keep = [i for i in range(module.num_features) if i not in set(spec["idxs"])]
            index = torch.tensor(keep, dtype=torch.long)
            module.weight = _slice_param(module.weight, keep, dim=0)
            module.bias = _slice_param(module.bias, keep, dim=0)
            if module.running_mean is not None:
                module.running_mean = module.running_mean.index_select(0, index).clone()
                module.running_var = module.running_var.index_select(0, index).clone()
            module.num_features = len(keep)
        elif kind == "linear_in":
            keep = [i for i in range(module.in_features) if i not in set(spec["idxs"])]
            module.weight = _slice_param(module.weight, keep, dim=1)
            module.in_features = len(keep)


# --------------------------------------------------------------------------
# Public resolver
# --------------------------------------------------------------------------

@dataclass
class Resolution:
    """How to prune ``layer`` at ``idxs``, and what it deletes per BatchNorm."""

    engine: str
    layer: str
    idxs: List[int]
    bn_deletions: Dict[str, Set[int]]
    _plan: object = None

    def prune_copy(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        pruned = copy.deepcopy(model)
        pruned.eval()
        if self.engine == "torch-pruning":
            _tp_prune(pruned, self.layer, self.idxs, example_input)
        else:
            _apply_plan(pruned, self._plan)
            with torch.no_grad():
                pruned(example_input)
        return pruned


def resolve(
    model: nn.Module, layer_name: str, idxs: Sequence[int], example_input: torch.Tensor
) -> Resolution:
    """Resolve pruning dependencies, preferring the general graph engine."""
    idxs = sorted(int(i) for i in idxs)
    try:
        scratch = copy.deepcopy(model)
        scratch.eval()
        # NB: no torch.no_grad() here -- torch-pruning traces the graph
        # through autograd and needs grad_fn to be recorded.
        bn_deletions = _tp_prune(scratch, layer_name, idxs, example_input)
        return Resolution("torch-pruning", layer_name, idxs, bn_deletions)
    except Exception:
        pass
    try:
        plan = _sequential_plan(model, layer_name, idxs, example_input)
        scratch = copy.deepcopy(model)
        scratch.eval()
        _apply_plan(scratch, plan)
        if not _structurally_consistent(scratch):
            raise UnsupportedGraphError("Sequential plan produced an inconsistent model.")
        with torch.no_grad():
            scratch(example_input)
        bn_deletions = {
            name: set(spec["idxs"]) for name, kind, spec in plan if kind == "bn"
        }
        return Resolution("sequential", layer_name, idxs, bn_deletions, _plan=plan)
    except UnsupportedGraphError:
        raise
    except Exception as err:
        raise UnsupportedGraphError(
            f"Could not resolve pruning dependencies for {layer_name!r}: {err}"
        ) from err
