import pytest
import torch

from prunecal import BudgetUnreachableError, compress, count_macs
from prunecal.data import synthetic_motor_imagery
from prunecal.models import ControlDense, EEGNet


@pytest.fixture()
def eegnet_and_data():
    torch.manual_seed(0)
    model = EEGNet(n_channels=3, n_samples=512, n_classes=2, F1=8, D=2, F2=16)
    data, labels = synthetic_motor_imagery(
        n_trials=64, generator=torch.Generator().manual_seed(0)
    )
    return model, data, labels


def test_budget_is_met_with_smallest_sufficient_pruning(eegnet_and_data):
    model, data, labels = eegnet_and_data
    pruned, report = compress(
        model, data, labels, layer="conv_temporal", budget=2.0
    )
    assert report.macs_after <= report.macs_before / 2.0
    # Minimality: one fewer pruned filter would miss the budget.
    if len(report.pruned) > 1:
        smaller, smaller_report = compress(
            model,
            data,
            labels,
            layer="conv_temporal",
            ratio=(len(report.pruned) - 1) / 8,
        )
        assert smaller_report.macs_after > report.macs_before / 2.0


def test_pruning_is_physical_and_model_still_runs(eegnet_and_data):
    model, data, labels = eegnet_and_data
    pruned, report = compress(
        model, data, labels, layer="conv_temporal", ratio=0.875
    )
    kept = 8 - round(8 * 0.875)
    assert pruned.conv_temporal.weight.shape[0] == kept
    assert pruned.bn_temporal.num_features == kept
    assert pruned.conv_spatial.weight.shape[0] == kept * 2  # depth multiplier D=2
    out = pruned(data[:4])
    assert out.shape == (4, 2)
    # The original model is untouched.
    assert model.conv_temporal.weight.shape[0] == 8


def test_report_names_the_layer_to_recalibrate(eegnet_and_data):
    model, data, labels = eegnet_and_data
    _, report = compress(model, data, labels, layer="conv_temporal", budget=2.0)
    assert report.recalibrate == ["bn_separable"]
    assert report.probe.n_batchnorms == 3


def test_lowest_scoring_filters_are_pruned(eegnet_and_data):
    model, data, labels = eegnet_and_data
    _, report = compress(
        model, data, labels, layer="conv_temporal", ratio=0.5
    )
    pruned_scores = [report.scores[i] for i in report.pruned]
    kept_scores = [report.scores[i] for i in report.kept]
    assert max(pruned_scores) <= min(kept_scores)


def test_unreachable_budget_raises():
    torch.manual_seed(0)
    model = ControlDense()
    x = torch.randn(4, 3, 64)
    with pytest.raises(BudgetUnreachableError):
        compress(model, layer="conv1", budget=3.0, criterion="magnitude", example_input=x)


def test_macs_counting_matches_hand_calculation():
    conv = torch.nn.Conv1d(3, 8, 5, padding=2, bias=False)
    total, per_layer = count_macs(conv, torch.randn(1, 3, 100))
    assert total == 8 * 100 * 3 * 5

    linear = torch.nn.Linear(16, 4)
    total, _ = count_macs(linear, torch.randn(1, 16))
    assert total == 16 * 4


def test_exactly_one_of_budget_or_ratio(eegnet_and_data):
    model, data, labels = eegnet_and_data
    with pytest.raises(ValueError):
        compress(model, data, labels, layer="conv_temporal")
    with pytest.raises(ValueError):
        compress(model, data, labels, layer="conv_temporal", budget=2.0, ratio=0.5)


def test_surgery_matches_masked_model(eegnet_and_data):
    """The report's own validation of the surgery (Section 3): physically
    removing filters must give the same outputs as a masked model in which
    the removed channels are prevented from contributing downstream (here:
    zeroing the pointwise convolution's input columns for the deleted
    channels)."""
    import copy

    model, data, labels = eegnet_and_data
    pruned, report = compress(model, data, labels, layer="conv_temporal", ratio=0.5)

    masked = copy.deepcopy(model).eval()
    D = 2
    dead_pointwise_inputs = [f * D + d for f in report.pruned for d in range(D)]
    with torch.no_grad():
        masked.conv_separable_point.weight[:, dead_pointwise_inputs] = 0.0
        out_masked = masked(data[:8])
        out_pruned = pruned(data[:8])
    torch.testing.assert_close(out_pruned, out_masked, rtol=1e-4, atol=1e-5)


def test_dependency_resolver_engines():
    """torch-pruning handles branching graphs; the exact sequential engine
    covers the grouped-with-depth-multiplier pattern (EEGNet's spatial
    convolution) that torch-pruning resolves incorrectly."""
    from prunecal._graph import resolve
    from prunecal.models import ControlSkip

    x1d = torch.randn(2, 3, 64)
    skip = resolve(ControlSkip(), "conv1", [0], x1d)
    assert skip.engine == "torch-pruning"

    eeg = EEGNet(n_channels=3, n_samples=512, n_classes=2, F1=8, D=2, F2=16)
    res = resolve(eeg, "conv_temporal", [0, 1], torch.randn(1, 1, 3, 512))
    assert res.engine == "sequential"
    assert res.bn_deletions == {
        "bn_temporal": {0, 1},
        "bn_spatial": {0, 1, 2, 3},
    }
