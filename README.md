# prunecal — prune, probe, recalibrate

A calibration-free compression pipeline for EEG decoders (and other BatchNorm CNNs), built for deployment on microcontrollers.

This is the reference implementation of the method described in the research report *"Compression pipeline for a calibration-free EEG decoder running on a microcontroller"* (マイクロコントローラ上で動作する較正不要脳波デコーダの圧縮パイプラインの提案). The report's three findings, which this package operationalizes:

1. **Compression and test-time adaptation do not compete.** The accuracy gained by label-free adaptation *grows* with the pruning ratio (+0.019 unpruned → +0.101 at 87.5 % pruning on BCI Competition IV 2b).
2. **The reason is that pruning damages the model's own BatchNorm records.** Removing filters changes the distributions flowing through the network, but the stored running statistics still describe the unpruned model. A separation experiment shows this mismatch (Δ, Eq. 3) is caused almost entirely by the pruning itself, not by the session change. Adaptation of a compressed model is therefore less "fitting a new session" and more "repairing self-inflicted damage" — damage that a single label-free pass fixes.
3. **Which layers need repair is decided by the wiring alone.** A data-free, training-free probe (zero one filter, compare pre-BatchNorm tensors exactly) identifies the affected layers in seconds. For EEGNet it names exactly one of three BatchNorm layers, and recalibrating that one layer matches full-network AdaBN (0.773 at the 87.5 % / int8 operating point, from label-free 16 trials).

## Install

```bash
pip install git+https://github.com/t0mat0ojam/prunecal.git
# or, for development:
git clone https://github.com/t0mat0ojam/prunecal.git && cd prunecal && pip install -e ".[test]"
```

Dependencies: `torch`, `torch-pruning`, `numpy`. Pruning dependencies (which downstream channels a filter removal deletes) are resolved by torch-pruning's dependency graph where its result validates, with an exact built-in propagation engine for sequential chains as the fallback — torch-pruning 1.6 mishandles grouped convolutions with a depth multiplier, which is precisely EEGNet's spatial convolution, so the fallback is what handles EEGNet itself (covered by tests).

## Quickstart

```python
import torch
from prunecal import compress, recalibrate, delta
from prunecal.models import EEGNet
from prunecal.data import synthetic_motor_imagery

# Your trained model and training data go here; this uses stand-ins.
train_x, train_y = synthetic_motor_imagery(n_trials=128)
model = EEGNet(n_channels=3, n_samples=512, n_classes=2)
# ... train `model` as usual ...

# 1) Compress to a MAC budget (or ratio=0.875 for a fixed pruning ratio).
pruned, report = compress(
    model, train_x, train_y,
    layer="conv_temporal",   # the pruning target (report, Section 3)
    budget=2.0,              # halve the multiply–accumulate count, Eq. 2
)
print(report.summary())
# -> pruned k/N filters by discriminability; MACs before -> after;
#    recalibrate: ['bn_separable']  (1/3 BatchNorm layers)

# 2) Deploy. When target-session data appears, repair the named layers
#    from unlabeled trials — one forward pass, no labels, no backprop.
target_x, _ = synthetic_motor_imagery(n_trials=16, session_shift=0.5)
recalibrate(pruned, target_x, layers=report.recalibrate)

# Optional diagnostics: the normalization mismatch per layer (Eq. 3).
print(delta(pruned, target_x))
```

`compress` never retrains: post-pruning fine-tuning would let surviving filters relearn the deleted ones and mask the effect being studied (report, Section 1).

## Run the probe on your own architecture

The probe is model-agnostic and needs no data or training — random weights give the same answer, because the answer is determined by the network's wiring:

```python
import torch
from prunecal import probe
from my_models import SomePublishedEMGNet   # any nn.Module with BatchNorm

model = SomePublishedEMGNet()
result = probe(model, layer="conv1", example_input=torch.randn(1, 8, 400))
print(result.summary())
# probe of 'conv1' (filter 0): 5 / 6 BatchNorm layers reached
#   reached   ...
#   untouched ...
```

`result.reached` is the set of BatchNorm layers whose statistics that pruning would invalidate — i.e. the layers a deployed device must recalibrate. The report's survey (Table 2) found that a *single-layer* repair set is the exception, not the rule: it requires a depthwise-separable structure **and** no shortcut past the channel-mixing operation. EEGNet qualifies (1/3); Deep4Net (4/4), ResNet-18 (19/20) and MobileNetV2 (50/52, despite its depthwise structure — the residual connections carry the perturbation past the mixing) do not. Running the probe on further EMG/ECG/wearable architectures takes seconds per model and directly extends that table.

The three control architectures used to validate the probe in the report (depthwise-only → 1/3, dense → 2/3, skip → 2/4) ship in `prunecal.models` and are exercised by the test suite.

## How the code maps to the report

| Report | Code |
|---|---|
| §2, Eq. 1 — discriminability criterion (ANOVA F on log band power) | `prunecal.criteria.discriminability_scores` (baselines: `magnitude_scores`, `random_scores`) |
| §3, Eq. 2 — MAC-budget solve, "unreachable" layers | `compress(..., budget=R)`; `BudgetUnreachableError`; `prunecal.macs.count_macs` |
| §3 — structured (physical) filter removal | `compress` internals (`prunecal._graph`: torch-pruning's dependency groups where they validate, an exact chain propagation for the grouped/depthwise EEGNet wiring they mishandle) |
| §4, Eq. 3 — normalization mismatch Δ | `prunecal.delta` |
| §4 — separation experiment (pruning vs. session as the source of Δ) | reproduced in spirit by `tests/test_recalibrate.py::test_pruning_itself_invalidates_only_the_probe_set` |
| §5, Fig. 5 — the probe | `prunecal.probe` |
| §5 — probe-set recalibration ≙ full AdaBN | `recalibrate(model, trials, layers=report.recalibrate)` vs. `layers=None` |

## Reproducing the report's numbers

The numbers quoted above come from the report's experiments on BCI Competition IV 2a/2b (9 subjects each, cross-session evaluation, 3 seeds), with per-session standardization and the training configuration described there; they are **not** produced by this repository's synthetic examples. To reproduce them, load 2a/2b via [MOABB](https://moabb.neurotechx.com/), train EEGNet / ShallowConvNet / Deep4Net as described in the report, and drive the training scripts through this package's API (`compress` → `recalibrate`). The experiment scripts are being migrated to import this module and will be added under `experiments/`.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite checks, among other things, that the probe reproduces the wiring-determined results of the report's Table 2 controls and EEGNet row; that the budget search returns the smallest sufficient pruning; that pruning is physical (tensor shapes shrink) and the original model is untouched; that Δ is driven to ~0 by recalibration; and the report's central mechanism — that on same-session data, pruning alone inflates Δ at exactly the layer the probe names, and recalibrating only that layer repairs it.

## Scope and limitations

- The single-layer result is architecture-conditional (depthwise-separable, no shortcut past the mixing); the probe tells you in seconds whether your model qualifies.
- `recalibrate` targets BatchNorm layers; models using LayerNorm/GroupNorm (which normalize per-sample, not from stored statistics) are out of scope.
- The report's session-shift findings hold *after* per-session standardization of the input; the pipeline does not remove the need for that preprocessing.
- Quantization: the report combines 87.5 % pruning with per-channel int8 weight quantization (37.8 kB → 3.1 kB) and finds the probe set remains sufficient and the order (quantize ↔ recalibrate) does not matter. Deployment-format conversion is toolchain-specific and left to the deployment scripts.

## Citation

If you use this code, please cite the report (see `CITATION.cff`).

## License

MIT — see `LICENSE`.
