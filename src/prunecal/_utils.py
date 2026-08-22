"""Small shared helpers."""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

LayerRef = Union[str, nn.Module]

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def resolve_layer(model: nn.Module, layer: LayerRef) -> Tuple[str, nn.Module]:
    """Resolve a layer given by qualified name or by module object.

    Returns ``(qualified_name, module)``. Raises ``KeyError``/``ValueError``
    with the list of available candidates if it cannot be found.
    """
    named = dict(model.named_modules())
    if isinstance(layer, str):
        if layer not in named:
            convs = [n for n, m in named.items() if isinstance(m, _CONV_TYPES)]
            raise KeyError(
                f"No module named {layer!r} in model. "
                f"Convolution layers available: {convs}"
            )
        return layer, named[layer]
    for name, module in named.items():
        if module is layer:
            return name, module
    raise ValueError(
        "The given module object is not part of this model. "
        "If you deep-copied the model, pass the layer's qualified name instead."
    )


def batchnorms(model: nn.Module) -> List[Tuple[str, _BatchNorm]]:
    """All BatchNorm layers of the model, in forward-definition order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _BatchNorm)]


def select_batchnorms(
    model: nn.Module, layers: Union[None, Iterable[LayerRef]]
) -> List[Tuple[str, _BatchNorm]]:
    """Select BatchNorm layers by name/module; ``None`` selects all of them."""
    all_bns = batchnorms(model)
    if layers is None:
        return all_bns
    wanted = set()
    for ref in layers:
        name, module = resolve_layer(model, ref)
        if not isinstance(module, _BatchNorm):
            raise TypeError(f"{name!r} is not a BatchNorm layer.")
        wanted.add(name)
    return [(n, m) for n, m in all_bns if n in wanted]
