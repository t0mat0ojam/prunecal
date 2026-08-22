"""Run the probe on any BatchNorm CNN -- no data, no training, seconds per model.

This reproduces the control rows of the report's Table 2, and shows the
pattern for extending the survey to further architectures (EMG, ECG,
wearable sensing, vision). To add a model of your own:

    result = probe(model, layer=<conv to prune>, example_input=<one input>)
    print(result.summary())

`result.reached` is the set of BatchNorm layers a deployed device would
need to recalibrate after pruning that layer. Random weights are fine:
the answer is determined by the wiring, so published architecture code
can be probed without any checkpoint.
"""

import torch

from prunecal import probe
from prunecal.models import ControlDense, ControlDepthwise, ControlSkip, EEGNet

torch.manual_seed(0)  # reproducible survey output

SURVEY = [
    ("depthwise-only control", ControlDepthwise(), "conv1", torch.randn(1, 3, 64)),
    ("dense control", ControlDense(), "conv1", torch.randn(1, 3, 64)),
    ("skip-connection control", ControlSkip(), "conv1", torch.randn(1, 3, 64)),
    ("EEGNet", EEGNet(3, 512, 2), "conv_temporal", torch.randn(1, 1, 3, 512)),
    # Add published EMG/ECG/vision models here, e.g.:
    # ("SomeEMGNet", SomeEMGNet(), "conv1", torch.randn(1, 8, 400)),
]


def main():
    print(f"{'model':<26} {'BN layers':>9} {'reached':>8}")
    for name, model, layer, x in SURVEY:
        result = probe(model, layer, x)
        print(f"{name:<26} {result.n_batchnorms:>9} {len(result.reached):>8}")


if __name__ == "__main__":
    main()
