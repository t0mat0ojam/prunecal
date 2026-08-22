import copy

import torch

from prunecal import compress, delta, recalibrate
from prunecal.data import synthetic_motor_imagery
from prunecal.models import EEGNet


def _model_with_source_stats(source_data):
    """EEGNet whose BN running statistics exactly reflect the source data
    (single-pass estimate with dropout off, i.e. `recalibrate` on all
    layers -- whose correctness the first test verifies independently)."""
    torch.manual_seed(0)
    model = EEGNet(n_channels=3, n_samples=512, n_classes=2, F1=8, D=2, F2=16)
    recalibrate(model, source_data)
    model.eval()
    return model


def test_recalibration_zeroes_delta():
    g = torch.Generator().manual_seed(0)
    source, _ = synthetic_motor_imagery(n_trials=64, generator=g)
    target, _ = synthetic_motor_imagery(n_trials=16, session_shift=0.5, generator=g)
    model = _model_with_source_stats(source)

    before = delta(model, target)
    assert before["bn_temporal"] > 0.05  # the session shift is visible

    recalibrate(model, target)  # layers=None -> full AdaBN
    after = delta(model, target)
    assert all(v < 5e-3 for v in after.values())


def test_only_selected_layers_are_touched():
    g = torch.Generator().manual_seed(1)
    source, _ = synthetic_motor_imagery(n_trials=64, generator=g)
    target, _ = synthetic_motor_imagery(n_trials=16, session_shift=0.5, generator=g)
    model = _model_with_source_stats(source)

    untouched_mean = model.bn_temporal.running_mean.clone()
    untouched_var = model.bn_temporal.running_var.clone()
    recalibrate(model, target, layers=["bn_separable"])
    assert torch.equal(model.bn_temporal.running_mean, untouched_mean)
    assert torch.equal(model.bn_temporal.running_var, untouched_var)
    assert delta(model, target, layers=["bn_separable"])["bn_separable"] < 5e-3


def test_pruning_itself_invalidates_only_the_probe_set():
    """The separation result of the report (Section 4): evaluated on data
    from the *same* session, pruning alone drives up Delta -- and only at
    the layer the probe names. Recalibrating that one layer repairs it."""
    g = torch.Generator().manual_seed(2)
    source, labels = synthetic_motor_imagery(n_trials=64, generator=g)
    model = _model_with_source_stats(source)

    baseline = delta(model, source)
    assert all(v < 5e-3 for v in baseline.values())  # stats match the source

    pruned, report = compress(
        model, source, labels, layer="conv_temporal", ratio=0.5
    )
    assert report.recalibrate == ["bn_separable"]

    after_pruning = delta(pruned, source)  # no session change at all
    assert after_pruning["bn_separable"] > 0.05
    assert after_pruning["bn_separable"] > 5 * after_pruning["bn_temporal"]
    assert after_pruning["bn_separable"] > 5 * after_pruning["bn_spatial"]

    calibration = source[:16]  # 16 unlabeled trials, as in the report
    recalibrate(pruned, calibration, layers=report.recalibrate)
    assert delta(pruned, calibration)["bn_separable"] < 5e-3


def test_probe_set_recalibration_tracks_full_adabn_after_pruning():
    """After pruning, the probe-set recalibration and full AdaBN estimate
    the invalidated layer from the same trials; their outputs may differ
    only through the untouched layers' (still valid) statistics."""
    g = torch.Generator().manual_seed(3)
    source, labels = synthetic_motor_imagery(n_trials=64, generator=g)
    model = _model_with_source_stats(source)
    pruned, report = compress(
        model, source, labels, layer="conv_temporal", ratio=0.5
    )
    calibration = source[:16]

    probe_set = recalibrate(copy.deepcopy(pruned), calibration, layers=report.recalibrate)
    full = recalibrate(copy.deepcopy(pruned), calibration, layers=None)

    with torch.no_grad():
        disagreement = (probe_set(calibration) - full(calibration)).abs().max()
        repair_size = (probe_set(calibration) - pruned(calibration)).abs().max()
    assert disagreement < 0.2 * repair_size
