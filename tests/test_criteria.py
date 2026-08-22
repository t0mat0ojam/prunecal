import torch
import torch.nn as nn

from prunecal import discriminability_scores, magnitude_scores, random_scores


def _band_kernel(freq_hz: float, length: int = 64, fs: float = 128.0) -> torch.Tensor:
    t = torch.arange(length) / fs
    kernel = torch.sin(2 * torch.pi * freq_hz * t) * torch.hann_window(length)
    return kernel / kernel.norm()


class TwoFilterNet(nn.Module):
    """One temporal convolution with two hand-crafted band filters."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, (1, 64), bias=False)
        with torch.no_grad():
            self.conv.weight[0, 0, 0] = _band_kernel(11.0)  # mu band
            self.conv.weight[1, 0, 0] = _band_kernel(31.0)  # unrelated band

    def forward(self, x):
        return self.conv(x).pow(2).mean(dim=(2, 3))


def _data_with_class_dependent_mu_power(n_trials=64, n_samples=512, fs=128.0):
    g = torch.Generator().manual_seed(0)
    labels = torch.arange(n_trials) % 2
    t = torch.arange(n_samples) / fs
    mu = torch.sin(2 * torch.pi * 11.0 * t + 2 * torch.pi * torch.rand(n_trials, 1, generator=g))
    other = torch.sin(2 * torch.pi * 31.0 * t + 2 * torch.pi * torch.rand(n_trials, 1, generator=g))
    amplitude = torch.where(labels == 0, 2.0, 0.4).unsqueeze(1)
    x = amplitude * mu + other + 0.3 * torch.randn(n_trials, n_samples, generator=g)
    return x.reshape(n_trials, 1, 1, n_samples), labels


def test_discriminability_prefers_the_class_informative_filter():
    """Eq. 1 must score the mu-band filter (whose output power differs by
    class) far above the 31 Hz filter (whose output power does not)."""
    model = TwoFilterNet()
    data, labels = _data_with_class_dependent_mu_power()
    scores = discriminability_scores(model, "conv", data, labels)
    assert scores.shape == (2,)
    assert scores[0] > 10 * scores[1]


def test_baseline_criteria_shapes():
    model = TwoFilterNet()
    assert magnitude_scores(model, "conv").shape == (2,)
    r = random_scores(model, "conv", generator=torch.Generator().manual_seed(0))
    assert r.shape == (2,)
    assert (0 <= r).all() and (r <= 1).all()
