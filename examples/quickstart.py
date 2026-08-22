"""End-to-end pipeline on synthetic data: train -> compress -> deploy -> recalibrate.

This runs in under a minute on CPU and demonstrates the API and the
mechanism (pruning inflates Delta at exactly the layer the probe names;
one label-free pass repairs it). It does NOT reproduce the report's
numbers -- for that, use BCI Competition IV 2a/2b via MOABB with the
training configuration described in the report.
"""

import torch

from prunecal import compress, delta, recalibrate
from prunecal.data import standardize_session, synthetic_motor_imagery
from prunecal.models import EEGNet


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    # "Source session": train the dense model. As in the report, every
    # session is standardized to mean 0 / std 1 before the network.
    train_x, train_y = synthetic_motor_imagery(n_trials=256, generator=g)
    test_x, test_y = synthetic_motor_imagery(n_trials=128, generator=g)
    train_x, test_x = standardize_session(train_x), standardize_session(test_x)
    model = EEGNet(n_channels=3, n_samples=512, n_classes=2, F1=8, D=2, F2=16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(30):
        for start in range(0, len(train_x), 64):
            xb, yb = train_x[start : start + 64], train_y[start : start + 64]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    print(f"dense model            : accuracy {accuracy(model, test_x, test_y):.3f}")

    # Compress the temporal convolution. (The report's operating point is
    # ratio=0.875 with F1=32; this toy model has F1=8, so 0.75 -- keeping
    # 2 of 8 filters -- is the comparable "almost too small" regime.)
    pruned, report = compress(
        model, train_x, train_y, layer="conv_temporal", ratio=0.75
    )
    print(report.summary())

    # "Target session": a distribution shift, unlabeled 16 trials available.
    target_x, target_y = synthetic_motor_imagery(
        n_trials=128, session_shift=0.4, generator=g
    )
    target_x = standardize_session(target_x)  # removes the input-level shift
    calibration = target_x[:16]

    print(f"pruned, no adaptation  : accuracy {accuracy(pruned, target_x, target_y):.3f}")
    print(f"  Delta per layer      : { {k: round(v, 3) for k, v in delta(pruned, calibration).items()} }")

    recalibrate(pruned, calibration, layers=report.recalibrate)
    print(f"pruned + recalibrated  : accuracy {accuracy(pruned, target_x, target_y):.3f}")
    print(f"  Delta per layer      : { {k: round(v, 3) for k, v in delta(pruned, calibration).items()} }")


if __name__ == "__main__":
    main()
