"""prunecal: prune -> probe -> recalibrate.

A calibration-free compression pipeline for EEG decoders (and other
BatchNorm CNNs), implementing the method described in the accompanying
research report:

1. ``compress(model, data, labels, ...)`` prunes the dominant convolution
   using a discriminability criterion (a one-way ANOVA F-statistic on
   log-power activations) down to a MAC budget or a fixed ratio, and
   returns the pruned model together with a report that names exactly
   which BatchNorm layers must be recalibrated.
2. ``probe(model, layer, example_input)`` determines, from the network
   wiring alone (no data, no training), which normalization layers are
   reached by pruning a given layer.
3. ``recalibrate(model, unlabeled_trials, layers=...)`` re-estimates the
   BatchNorm running statistics of the named layers from a single pass
   over unlabeled target-session trials (AdaBN restricted to the probe
   set). No labels, no backpropagation.
4. ``delta(model, data)`` measures the normalization mismatch
   Delta = ((mu_t - mu_s) / sigma_s)^2 + (sigma_t / sigma_s - 1)^2
   per BatchNorm layer (Eq. 3 of the report).
"""

from .criteria import discriminability_scores, magnitude_scores, random_scores
from .macs import count_macs
from .probe import ProbeResult, probe
from .pruning import BudgetUnreachableError, CompressionReport, compress
from .recalibrate import delta, recalibrate

__version__ = "0.1.0"

__all__ = [
    "compress",
    "recalibrate",
    "probe",
    "delta",
    "count_macs",
    "discriminability_scores",
    "magnitude_scores",
    "random_scores",
    "CompressionReport",
    "ProbeResult",
    "BudgetUnreachableError",
    "__version__",
]
