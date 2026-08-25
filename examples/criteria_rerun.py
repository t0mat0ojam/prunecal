"""Criterion comparison, canonical rerun -- BCI IV 2b AND 2a, temporal axis.

Supersedes the sweep/fig4 lineage. What changed and why, so the paper's
methods section can be written from this docstring:

  AXIS      temporal convolution (the axis every other experiment prunes),
            not the pointwise/F2 axis of the original sweeps.
  CONFIG    F1=32, D=2, F2=64 -- the configuration the paper states.
  PRUNING   physical (prunecal.compress), not masking. No fine-tuning.
  RECAL     because temporal pruning invalidates bn3 differently per
            criterion, every arm is recalibrated on TRAINING data
            (probe-computed set) before evaluation: pure repair of
            pruning-induced damage, zero target-session information.
            The un-recalibrated accuracy is logged as acc_raw.
  GRID      sparsity 0, .2, .4, .6, .8, .875  (87.5% keeps 4 of 32).
  SPLIT     2b: LOSO over the three feedback sessions
            ("2train","3test","4test"); 2a: LOSO over its two sessions.
  ARMS      random | magnitude | disc_eq1 (paper Eq.1: ANOVA F on log
            band power) | stability_cv (paper Eq.: inverse CV of
            per-session mean activation) | profile_diff (the legacy
            'within_session' score from pilot.py, ported to this axis;
            extended to K classes via mean pairwise profile distance)
            | combined_rank (disc + stability rank average).

Run:  pip install moabb git+https://github.com/USER/prunecal.git
      python criteria_rerun.py                      # full: both datasets
      DATASETS=2b SUBJECTS=1 SEEDS=1 EPOCHS=5 python criteria_rerun.py
      MOCK=1 python criteria_rerun.py               # no-download smoke test
Resumable: reruns skip (dataset, subject, seed, fold) cells already in the CSV.
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
import torch

from prunecal import compress, magnitude_scores, random_scores, recalibrate
from prunecal.criteria import discriminability_scores
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

DATASETS   = os.environ.get("DATASETS", "2b,2a").split(",")
SUBJECTS   = [int(s) for s in os.environ.get("SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
SEEDS      = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS_2B  = int(os.environ.get("EPOCHS", "60"))
EPOCHS_2A  = int(os.environ.get("EPOCHS_2A", os.environ.get("EPOCHS", "200")))
SPARSITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 0.875]
ARMS       = ["random", "magnitude", "disc_eq1", "stability_cv",
              "profile_diff", "combined_rank"]
N_CALIB_NOTE = "recal uses TRAINING data only in this experiment"
FEEDBACK_SESSIONS_2B = ["2train", "3test", "4test"]
OUT        = os.environ.get("OUT", "criteria_rerun.csv")
DEV        = "cuda" if torch.cuda.is_available() else "cpu"
N_TIME_BINS = 8   # legacy profile score, kept identical to pilot.py


# ------------------------------------------------------------------- data
def load_subject(dataset, subject):
    if os.environ.get("MOCK", "0") == "1":
        return load_mock(dataset, subject)
    import mne, moabb
    mne.set_log_level("ERROR"); moabb.set_log_level("error")
    from moabb.datasets import BNCI2014_001, BNCI2014_004
    from moabb.paradigms import MotorImagery
    if dataset == "2b":
        paradigm = MotorImagery(n_classes=2, fmin=4, fmax=38, resample=128)
        X, y, meta = paradigm.get_data(BNCI2014_004(), subjects=[subject])
        keep = meta.session.isin(FEEDBACK_SESSIONS_2B).to_numpy()
        X, y, sess = X[keep], np.asarray(y)[keep], meta.session.to_numpy()[keep]
    else:
        paradigm = MotorImagery(n_classes=4, fmin=4, fmax=38, resample=128)
        X, y, meta = paradigm.get_data(BNCI2014_001(), subjects=[subject])
        sess = meta.session.to_numpy()
    names = sorted(set(np.asarray(y)))
    lut = {n: i for i, n in enumerate(names)}
    y = np.array([lut[v] for v in np.asarray(y)], dtype="int64")
    T = (X.shape[-1] // 32) * 32
    return {s: (X[sess == s, :, :T].astype("float32"), y[sess == s])
            for s in dict.fromkeys(sess)}          # preserves session order


def load_mock(dataset, subject):
    rng = np.random.default_rng(subject)
    n_cls, C = (2, 3) if dataset == "2b" else (4, 22)
    names = [f"s{i}" for i in range(3 if dataset == "2b" else 2)]
    t = np.arange(512) / 128.0
    out = {}
    for s in names:
        n = 96
        y = (np.arange(n) % n_cls).astype("int64")
        X = rng.normal(0, 1, (n, C, 512)).astype("float32")
        for c in range(n_cls):
            X[y == c, c % C] += 1.3 * np.sin(2 * np.pi * (9 + 2 * c) * t)
        out[s] = (X, y)
    return out


def standardize(x):
    mu = x.mean(axis=(0, 2), keepdims=True)
    sd = x.std(axis=(0, 2), keepdims=True)
    return (x - mu) / (sd + 1e-8)


# ------------------------------------------------------------------- training
def train(model, X, y, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = torch.tensor(X[idx]).unsqueeze(1).to(DEV)
            yb = torch.tensor(y[idx]).to(DEV)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model.cpu().eval()


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    hits = 0
    for i in range(0, len(X), 128):
        xb = torch.tensor(X[i:i + 128]).unsqueeze(1)
        hits += (model(xb).argmax(1).numpy() == y[i:i + 128]).sum()
    return hits / len(y)


# ------------------------------------------------------------------- criteria
@torch.no_grad()
def _temporal_acts(model, X):
    """Per-trial activations of the temporal conv output: [N, F1, C, T]."""
    store = []
    h = model.conv_temporal.register_forward_hook(lambda m, i, o: store.append(o.cpu()))
    model.eval()
    for i in range(0, len(X), 128):
        model(torch.tensor(X[i:i + 128]).unsqueeze(1))
    h.remove()
    return torch.cat(store)


def stability_cv(model, sessions):
    """Paper stability: 1/(CV+eps) of per-session mean |activation| per filter."""
    means = []
    for X, _ in sessions:
        a = _temporal_acts(model, X).abs().mean(dim=(0, 2, 3))
        means.append(a.numpy())
    M = np.stack(means)                                   # [S, F1]
    eps = 1e-8
    cv = M.std(axis=0) / (np.abs(M.mean(axis=0)) + eps)
    return 1.0 / (cv + eps)


def profile_diff(model, sessions, n_bins=N_TIME_BINS):
    """Legacy pilot.py 'within_session' score, ported to the temporal axis and
    generalised to K classes as the mean pairwise |class-profile difference|."""
    scores = []
    for X, y in sessions:
        a = _temporal_acts(model, X).mean(dim=2).numpy()  # [N, F1, T] (mean over EEG ch)
        n, f1, t = a.shape
        binned = a[:, :, : (t // n_bins) * n_bins].reshape(n, f1, n_bins, -1).mean(-1)
        classes = sorted(set(y.tolist()))
        profs = [binned[y == c].mean(0) for c in classes]        # each [F1, bins]
        pair = [np.abs(profs[i] - profs[j]).mean(1)
                for i, j in itertools.combinations(range(len(classes)), 2)]
        scores.append(np.mean(pair, axis=0))
    return np.mean(scores, axis=0)


def scores_for(arm, model, Xtr, ytr, sessions, seed):
    x4 = torch.tensor(Xtr).unsqueeze(1)
    if arm == "random":
        return random_scores(model, "conv_temporal",
                             generator=torch.Generator().manual_seed(seed))
    if arm == "magnitude":
        return magnitude_scores(model, "conv_temporal")
    if arm == "disc_eq1":
        return discriminability_scores(model, "conv_temporal", x4, torch.tensor(ytr))
    if arm == "stability_cv":
        return torch.tensor(stability_cv(model, sessions), dtype=torch.float32)
    if arm == "profile_diff":
        return torch.tensor(profile_diff(model, sessions), dtype=torch.float32)
    if arm == "combined_rank":
        from scipy.stats import rankdata
        d = discriminability_scores(model, "conv_temporal", x4, torch.tensor(ytr)).numpy()
        s = stability_cv(model, sessions)
        return torch.tensor((rankdata(d) + rankdata(s)) / (2 * len(d)),
                            dtype=torch.float32)
    raise ValueError(arm)


# ------------------------------------------------------------------- run
def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.dataset, r.subject, r.seed, r.fold) for r in prev.itertuples()}
        print(f"resuming -- {len(done)} (dataset, subject, seed, fold) cells done")

    for dataset in DATASETS:
        epochs = EPOCHS_2B if dataset == "2b" else EPOCHS_2A
        for subject in SUBJECTS:
            data = {s: (standardize(X), y) for s, (X, y) in
                    load_subject(dataset, subject).items()}
            sess = list(data.keys())
            for seed in SEEDS:
                for held_out in sess:
                    if (dataset, subject, seed, held_out) in done:
                        continue
                    tr = [s for s in sess if s != held_out]
                    Xtr = np.concatenate([data[s][0] for s in tr])
                    ytr = np.concatenate([data[s][1] for s in tr])
                    Xte, yte = data[held_out]
                    n_cls = int(ytr.max()) + 1

                    dense = train(EEGNet(Xtr.shape[1], Xtr.shape[2], n_cls,
                                         F1=32, D=2, F2=64), Xtr, ytr, epochs, seed)
                    acc_dense = accuracy(dense, Xte, yte)
                    print(f"[{dataset} s{subject} seed{seed} fold {held_out}] "
                          f"dense={acc_dense:.3f}", flush=True)

                    rows = []
                    train_sessions = [data[s] for s in tr]
                    x4tr = torch.tensor(Xtr).unsqueeze(1)
                    for arm in ARMS:
                        sc = scores_for(arm, dense, Xtr, ytr, train_sessions, seed)
                        fn = lambda m, l, d, y_, _sc=sc: _sc     # fixed scores via callable
                        fn.__name__ = arm
                        for sp in SPARSITIES:
                            if sp == 0.0:
                                rows.append(dict(dataset=dataset, subject=subject,
                                                 seed=seed, fold=held_out, arm=arm,
                                                 sparsity=sp, acc=acc_dense,
                                                 acc_raw=acc_dense, dense=acc_dense))
                                continue
                            pruned, rep = compress(dense, x4tr, torch.tensor(ytr),
                                                   layer="conv_temporal", ratio=sp,
                                                   criterion=fn)
                            acc_raw = accuracy(pruned, Xte, yte)
                            if rep.recalibrate:            # repair on TRAINING data
                                recalibrate(pruned, x4tr, layers=rep.recalibrate)
                            rows.append(dict(dataset=dataset, subject=subject,
                                             seed=seed, fold=held_out, arm=arm,
                                             sparsity=sp,
                                             acc=accuracy(pruned, Xte, yte),
                                             acc_raw=acc_raw, dense=acc_dense))
                    pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                              header=not os.path.exists(OUT))

    # headline statistics at the primary endpoint
    df = pd.read_csv(OUT)
    from scipy.stats import wilcoxon
    print("\n=== 87.5% endpoint, mean over folds/seeds, Wilcoxon vs random ===")
    for dataset in sorted(df.dataset.unique()):
        d = df[(df.dataset == dataset) & (df.sparsity == 0.875)]
        per = d.groupby(["arm", "subject"]).acc.mean().unstack()   # arms x subjects
        if "random" not in per.index:
            continue
        for arm in per.index:
            if arm == "random":
                continue
            diff = (per.loc[arm] - per.loc["random"]).dropna()
            if len(diff) >= 5:
                p = wilcoxon(diff, alternative="two-sided").pvalue
                print(f"  [{dataset}] {arm:14s} vs random: "
                      f"{diff.mean():+.3f}  ({int((diff > 0).sum())}/{len(diff)}, p={p:.3f})")


if __name__ == "__main__":
    main()
