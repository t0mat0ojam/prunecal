"""Extend the report's architecture survey (Table 2 / 表6) to EMG and ECG models.

The probe needs no data and no training -- random weights give the same
answer, because which BatchNorm layers a pruning invalidates is determined
by the wiring alone. Each architecture below is therefore one probe() call.

Provenance matters for a paper table, so every row is tagged:
  [canonical]      third-party reference implementation (torchvision / tsai),
                   not reimplemented by us
  [reimplemented]  written here from the paper's description -- verify the
                   layer list against the authors' official code before citing
  [ours]           the report's own models

Two of the [canonical] rows (ResNet-18, MobileNetV2) already appear in the
report; they are included as a harness check and must reproduce 19/20 and
50/52. The script warns loudly if they do not.

Run:  pip install tsai            # torchvision ships with Colab already
      python probe_survey.py

Add a model of your own (e.g. from a paper's official GitHub repo): paste
the nn.Module class into this file, append one SURVEY entry with the input
shape its paper uses, and re-run. `first_dense_conv` picks the pruning
target automatically (the first groups=1 convolution, the analogue of
EEGNet's temporal layer); pass an explicit layer name to override.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prunecal import probe
from prunecal.models import EEGNet


def first_dense_conv(model: nn.Module) -> str:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.groups == 1:
            return name
    raise ValueError("No dense convolution found.")


class HannunECGResNet(nn.Module):
    """[reimplemented] 1D residual CNN of Hannun et al., Nature Medicine 2019
    (single-lead ECG arrhythmia detection). Written from the paper's
    description: k=16 convolutions, 16 residual blocks, channels doubling
    every 4 blocks, subsampling by 2 every other block. Simplifications to
    verify against the official code (github.com/awni/ecg): post-activation
    block order, and a strided 1x1 convolution on the shortcut where the
    original may zero-pad channels, and k=15 rather than the paper's k=16
    (odd kernels make symmetric padding exact; wiring is unaffected).
    """

    def __init__(self, n_classes: int = 4, base: int = 32, n_blocks: int = 16, k: int = 15):
        super().__init__()
        self.stem = nn.Conv1d(1, base, k, padding=k // 2, bias=False)
        self.stem_bn = nn.BatchNorm1d(base)
        blocks, shortcuts = [], []
        ch = base
        for i in range(n_blocks):
            out = base * (2 ** (i // 4))
            stride = 2 if i % 2 == 1 else 1
            blocks.append(nn.Sequential(
                nn.Conv1d(ch, out, k, stride=stride, padding=k // 2, bias=False),
                nn.BatchNorm1d(out),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(out, out, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out),
            ))
            shortcuts.append(
                nn.Identity() if (ch == out and stride == 1)
                else nn.Conv1d(ch, out, 1, stride=stride, bias=False)
            )
            ch = out
        self.blocks = nn.ModuleList(blocks)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = F.relu(block(x) + shortcut(x))
        return self.head(x.mean(dim=-1))


def survey_rows():
    rows = [
        ("EEGNet", "[ours]", lambda: EEGNet(3, 512, 2), torch.randn(1, 1, 3, 512), "conv_temporal"),
    ]
    try:
        import torchvision.models as tvm
        rows += [
            ("ResNet-18", "[canonical, harness check: report says 19/20]",
             lambda: tvm.resnet18(weights=None), torch.randn(1, 3, 64, 64), None),
            ("MobileNetV2", "[canonical, harness check: report says 50/52]",
             lambda: tvm.mobilenet_v2(weights=None), torch.randn(1, 3, 64, 64), None),
        ]
    except ImportError:
        print("torchvision not installed -- skipping ResNet-18 / MobileNetV2 checks")
    try:
        from tsai.models.XceptionTime import XceptionTime
        from tsai.models.InceptionTime import InceptionTime
        rows += [
            ("XceptionTime (sEMG)", "[canonical tsai impl of Rahimian et al.]",
             lambda: XceptionTime(8, 10), torch.randn(1, 8, 400), None),
            ("InceptionTime", "[canonical tsai impl of Ismail Fawaz et al.]",
             lambda: InceptionTime(8, 10), torch.randn(1, 8, 400), None),
        ]
    except ImportError:
        print("tsai not installed (pip install tsai) -- skipping XceptionTime / InceptionTime")
    rows.append(
        ("ECG ResNet (Hannun 2019)", "[reimplemented -- verify before citing]",
         lambda: HannunECGResNet(), torch.randn(1, 1, 1024), None)
    )
    return rows


def main():
    print(f"{'model':<26} {'BN':>4} {'reached':>8}   provenance")
    latex = []
    for name, note, build, x, layer in survey_rows():
        model = build().eval()
        layer = layer or first_dense_conv(model)
        try:
            result = probe(model, layer, x)
        except Exception as err:
            print(f"{name:<26} {'--':>4} {'error':>8}   {type(err).__name__}: {err}")
            continue
        n, r = result.n_batchnorms, len(result.reached)
        print(f"{name:<26} {n:>4} {r:>8}   {note}")
        latex.append(f"{name} & {n} & {r} \\\\")
        if "harness check" in note:
            expected = {"ResNet-18": (20, 19), "MobileNetV2": (52, 50)}[name]
            if (n, r) != expected:
                print(f"  !! WARNING: harness disagrees with the report {expected} -- investigate")
    print("\nLaTeX rows (model & #BN & #reached):")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
