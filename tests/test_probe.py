"""The probe must reproduce the wiring-determined results of the report.

Table 2 (control conditions):
    depthwise-only : 3 BatchNorm layers, 1 reached
    dense conv     : 3 BatchNorm layers, 2 reached
    skip connection: 4 BatchNorm layers, 2 reached
EEGNet             : 3 BatchNorm layers, 1 reached (bn3 only)

And the report's key structural claim: the answer depends only on the
wiring -- not on the weights, and not on which filter is zeroed.
"""

import torch

from prunecal import probe
from prunecal.models import ControlDense, ControlDepthwise, ControlSkip, EEGNet


def _x1d():
    return torch.randn(2, 3, 64, generator=torch.Generator().manual_seed(0))


def test_control_depthwise_reaches_one_of_three():
    result = probe(ControlDepthwise(), "conv1", _x1d())
    assert result.n_batchnorms == 3
    assert result.reached == ["bn3"]
    assert set(result.untouched) == {"bn1", "bn2"}


def test_control_dense_reaches_two_of_three():
    result = probe(ControlDense(), "conv1", _x1d())
    assert result.n_batchnorms == 3
    assert result.reached == ["bn2", "bn3"]
    assert result.untouched == ["bn1"]


def test_control_skip_reaches_two_of_four():
    result = probe(ControlSkip(), "conv1", _x1d())
    assert result.n_batchnorms == 4
    assert set(result.reached) == {"bn3", "bn4"}
    assert set(result.untouched) == {"bn1", "bn2"}


def test_eegnet_reaches_only_the_third_batchnorm():
    model = EEGNet(n_channels=3, n_samples=512, n_classes=2)
    x = torch.randn(2, 1, 3, 512, generator=torch.Generator().manual_seed(0))
    result = probe(model, "conv_temporal", x)
    assert result.n_batchnorms == 3
    assert result.reached == ["bn_separable"]
    assert set(result.untouched) == {"bn_temporal", "bn_spatial"}


def test_answer_is_wiring_determined():
    """Same result for any zeroed filter and any weight initialization."""
    x = torch.randn(2, 1, 3, 512, generator=torch.Generator().manual_seed(1))
    model = EEGNet(n_channels=3, n_samples=512, n_classes=2, F1=8)
    reference = probe(model, "conv_temporal", x, filter_index=0)
    for filter_index in (1, 7):
        other = probe(model, "conv_temporal", x, filter_index=filter_index)
        assert other.reached == reference.reached

    torch.manual_seed(12345)
    reinitialized = EEGNet(n_channels=3, n_samples=512, n_classes=2, F1=8)
    assert probe(reinitialized, "conv_temporal", x).reached == reference.reached


def test_probe_does_not_modify_the_model():
    model = ControlDense()
    before = model.conv1.weight.detach().clone()
    probe(model, "conv1", _x1d())
    assert torch.equal(model.conv1.weight, before)


def test_probe_is_robust_to_unlucky_initializations():
    # Regression test: a coincidentally dead ReLU channel (all-negative
    # pre-activations for one random init) used to gate the perturbation
    # to exactly zero and hide a reached layer. The wiring answer must be
    # identical across re-initializations.
    x = torch.randn(1, 3, 64, generator=torch.Generator().manual_seed(1))
    for seed in range(25):
        torch.manual_seed(seed)
        result = probe(ControlSkip(), "conv1", x)
        assert set(result.reached) == {"bn3", "bn4"}, f"seed {seed}"
