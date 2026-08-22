"""Synthetic two-class motor-imagery-like data.

For tests and examples only -- a stand-in with the same *shape* of
structure as motor imagery (class-dependent band-power lateralization in
the mu band on two "sensorimotor" channels, plus broadband noise), not a
substitute for real EEG. Reproduce the report's numbers with BCI
Competition IV 2a/2b via MOABB and your own training scripts.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def synthetic_motor_imagery(
    n_trials: int = 128,
    n_channels: int = 3,
    n_samples: int = 512,
    fs: float = 128.0,
    mu_hz: float = 11.0,
    snr: float = 1.0,
    session_shift: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(trials [N, 1, C, T], labels [N])`` with two classes.

    Class 0 attenuates the mu-band rhythm on channel 0 ("left-hemisphere
    desynchronization"), class 1 on the last channel. ``session_shift``
    adds a global affine change (scale and offset) imitating a
    between-session distribution shift *before* per-session
    standardization would remove it; use it to create a shifted "target
    session" for recalibration demos.
    """
    g = generator
    labels = torch.randint(0, 2, (n_trials,), generator=g)
    t = torch.arange(n_samples) / fs
    phase = 2 * torch.pi * torch.rand(n_trials, 1, generator=g)
    rhythm = torch.sin(2 * torch.pi * mu_hz * t + phase)  # [N, T]

    x = torch.randn(n_trials, n_channels, n_samples, generator=g)
    amp = torch.ones(n_trials, n_channels, 1) * snr
    amp[labels == 0, 0] *= 0.2   # ERD on channel 0 for class 0
    amp[labels == 1, -1] *= 0.2  # ERD on channel -1 for class 1
    x = x + amp * rhythm.unsqueeze(1)

    if session_shift:
        x = (1.0 + session_shift) * x + session_shift

    return x.unsqueeze(1), labels


def standardize_session(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-session, per-channel standardization to mean 0 / std 1.

    Mirrors the report's preprocessing (Section 1), which removes
    amplitude differences caused by electrode contact before the data
    reaches the network -- the reason the session-derived component of
    Delta is small in the report's separation experiment. Apply it to
    each session (source and target) independently.
    """
    dims = (0,) + tuple(range(2, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return (x - mean) / (std + eps)
