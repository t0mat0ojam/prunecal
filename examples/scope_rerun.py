"""EXP-04 -- 再較正すべき層の構造的決定 (regenerates 表validate).

EXP-03 already showed that recalibrating the probe-computed set matches
full AdaBN in all six dataset x architecture combinations. What it did
NOT show is that the probe *identifies* the right layers rather than
merely counting how many to repair. That is this experiment's job.

Arms (all use 16 UNLABELED target-session trials):
  none        no recalibration
  first_only  recalibrate only the FIRST BatchNorm layer. For EEGNet and
              Deep4/conv_3 the probe says that layer is NOT reached, so
              this arm should behave like `none`; for Shallow it IS the
              only layer, so it should equal `computed` by construction.
              The rule therefore makes opposite predictions in different
              architectures -- the paper's discriminating control.
  random_k    recalibrate k randomly chosen BatchNorm layers, where
              k = |computed set|. Same budget as `computed`, wrong
              layers (usually). Isolates "identity" from "count".
  computed    the probe-computed set
  all         every BatchNorm layer (full AdaBN)

For every arm we log accuracy AND Δ (Eq. 3) at the reached / untouched
layers afterwards. The mechanistic prediction: after `first_only` or
`random_k`, Δ at the reached layers stays high (the wrong layer was
repaired); after `computed` it is 0.

Scope: 2b (development set), three architectures, EXP-02 targets,
budget-matched at 1.5x / 2.0x / 3.0x. Protocol identical to EXP-01/02/03.
The architecture survey (表6) is data-free and lives in probe_survey.py.

Run:  MODELS=eegnet  OUT=scope_rerun.csv python scope_rerun.py   (~1.5 h)
      MODELS=shallow OUT=scope_rerun.csv python scope_rerun.py   (~1 h)
      MODELS=deep4   OUT=scope_rerun.csv python scope_rerun.py   (~3 h)
      MOCK=1 SUBJECTS=1 SEEDS=1 EPOCHS=2 OUT=scope_SMOKE.csv python scope_rerun.py
"""

import copy
import json
import os
import types
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from prunecal import BudgetUnreachableError, compress, delta, recalibrate
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

MODELS    = os.environ.get("MODELS", "eegnet,shallow,deep4").split(",")
SUBJECTS  = [int(s) for s in os.environ.get("SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
SEEDS     = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS    = int(os.environ.get("EPOCHS", "60"))
BUDGETS   = [1.5, 2.0, 3.0]
ARMS      = ["none", "first_only", "random_k", "computed", "all"]
N_CALIB   = 16
FEEDBACK_SESSIONS_2B = ["2train", "3test", "4test"]
OUT       = os.environ.get("OUT", "scope_rerun.csv")
DEV       = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS   = {"eegnet": "conv_temporal",
             "shallow": "conv_time_spat.conv_time",
             "deep4": "conv_3"}


# ------------------------------------------------------------------ data
def load_subject(subject):
    if os.environ.get("MOCK", "0") == "1":
        rng = np.random.default_rng(subject)
        t = np.arange(576) / 128.0
        out = {}
        for si, s in enumerate(["2train", "3test", "4test"]):
            y = (np.arange(128) % 2).astype("int64")
            X = rng.normal(0, 1, (128, 3, 576)).astype("float32")
            for c in (0, 1):
                X[y == c, c * 2] += 1.3 * np.sin(2 * np.pi * (9 + 2 * c) * t)
            out[s] = (X * (1 + 0.2 * si) + 0.1 * si, y)
        return out
    import mne, moabb
    mne.set_log_level("ERROR"); moabb.set_log_level("error")
    from moabb.datasets import BNCI2014_004
    from moabb.paradigms import MotorImagery
    paradigm = MotorImagery(n_classes=2, fmin=4, fmax=38, resample=128)
    X, y, meta = paradigm.get_data(BNCI2014_004(), subjects=[subject])
    keep = meta.session.isin(FEEDBACK_SESSIONS_2B).to_numpy()
    X, y, sess = X[keep], np.asarray(y)[keep], meta.session.to_numpy()[keep]
    lut = {n: i for i, n in enumerate(sorted(set(y)))}
    y = np.array([lut[v] for v in y], dtype="int64")
    T = (X.shape[-1] // 32) * 32
    return {s: (X[sess == s, :, :T].astype("float32"), y[sess == s])
            for s in dict.fromkeys(sess)}


def standardize(x):
    mu = x.mean(axis=(0, 2), keepdims=True)
    sd = x.std(axis=(0, 2), keepdims=True)
    return (x - mu) / (sd + 1e-8)


# ------------------------------------------------------------------ models
def _unfuse(model):
    for mod in model.modules():
        if type(mod).__name__ == "CombinedConv":
            mod.forward = types.MethodType(
                lambda self, x: self.conv_spat(self.conv_time(x)), mod)
    return model


def build_model(name, C, T, n_cls):
    if name == "eegnet":
        return EEGNet(C, T, n_cls, F1=32, D=2, F2=64), True
    from braindecode.models import Deep4Net, ShallowFBCSPNet
    cls = {"shallow": ShallowFBCSPNet, "deep4": Deep4Net}[name]
    return _unfuse(cls(n_chans=C, n_outputs=n_cls, n_times=T)), False


def shape(is4d, x):
    return x.unsqueeze(1) if is4d else x


def bn_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, _BatchNorm)]


def train(model, X, y, epochs, seed, is4d):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(
                model(shape(is4d, torch.tensor(X[idx])).to(DEV)),
                torch.tensor(y[idx]).to(DEV)).backward()
            opt.step()
    return model.cpu().eval()


@torch.no_grad()
def accuracy(model, X, y, is4d):
    model.eval()
    hits = 0
    for i in range(0, len(X), 128):
        hits += (model(shape(is4d, torch.tensor(X[i:i + 128]))).argmax(1).numpy()
                 == y[i:i + 128]).sum()
    return hits / len(y)


def arm_layers(arm, all_bn, computed, rng):
    if arm == "none":
        return []
    if arm == "first_only":
        return all_bn[:1]
    if arm == "random_k":
        k = len(computed)
        return [] if k == 0 else list(rng.choice(all_bn, size=min(k, len(all_bn)),
                                                 replace=False))
    if arm == "computed":
        return list(computed)
    if arm == "all":
        return list(all_bn)
    raise ValueError(arm)


def split_delta(d, reached):
    r = [v for k, v in d.items() if k in reached]
    u = [v for k, v in d.items() if k not in reached]
    return (float(np.mean(r)) if r else np.nan,
            float(np.mean(u)) if u else np.nan)


# ------------------------------------------------------------------ main
def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.model, r.subject, r.seed, r.fold) for r in prev.itertuples()}
        print(f"resuming -- {len(done)} cells done")

    for mname in MODELS:
        target = TARGETS[mname]
        for subject in SUBJECTS:
            data = {s: (standardize(X), y) for s, (X, y) in load_subject(subject).items()}
            sess = list(data.keys())
            for seed in SEEDS:
                for held_out in sess:
                    if (mname, subject, seed, held_out) in done:
                        continue
                    tr = [s for s in sess if s != held_out]
                    Xtr = np.concatenate([data[s][0] for s in tr])
                    ytr = np.concatenate([data[s][1] for s in tr])
                    Xtg, ytg = data[held_out]
                    Xcal, (Xev, yev) = Xtg[:N_CALIB], (Xtg[N_CALIB:], ytg[N_CALIB:])
                    n_cls = int(ytr.max()) + 1

                    model, is4d = build_model(mname, Xtr.shape[1], Xtr.shape[2], n_cls)
                    dense = train(model, Xtr, ytr, EPOCHS, seed, is4d)
                    x4tr = shape(is4d, torch.tensor(Xtr))
                    x4cal = shape(is4d, torch.tensor(Xcal))
                    all_bn = bn_names(dense)
                    acc_dense = accuracy(dense, Xev, yev, is4d)
                    print(f"[{mname} s{subject} seed{seed} fold {held_out}] "
                          f"dense={acc_dense:.3f} bn={len(all_bn)}", flush=True)

                    rows = []
                    for R in BUDGETS:
                        try:
                            pruned, rep = compress(dense, x4tr, torch.tensor(ytr),
                                                   layer=target, budget=R,
                                                   criterion="discriminability")
                        except BudgetUnreachableError:
                            rows.append(dict(model=mname, subject=subject, seed=seed,
                                             fold=held_out, layer=target, budget=R,
                                             reachable=False, arm="", acc=np.nan))
                            continue
                        computed = list(rep.recalibrate)
                        # independent draw per (subject, seed, fold, budget) so the
                        # random_k control is not the same wrong set everywhere
                        rng = np.random.default_rng(
                            abs(hash((mname, subject, seed, held_out, R))) % (2**32))
                        base = dict(model=mname, subject=subject, seed=seed,
                                    fold=held_out, layer=target, budget=R,
                                    reachable=True, acc_dense=round(acc_dense, 4),
                                    n_bn=len(all_bn), n_recal=len(computed),
                                    computed_layers=";".join(computed),
                                    sparsity=round(len(rep.pruned) /
                                                   (len(rep.pruned) + len(rep.kept)), 3),
                                    achieved=round(rep.macs_before / rep.macs_after, 3))
                        for arm in ARMS:
                            layers = arm_layers(arm, all_bn, computed, rng)
                            m = copy.deepcopy(pruned)
                            if layers:
                                recalibrate(m, x4cal, layers=layers)
                            d = delta(m, x4cal)
                            dr, du = split_delta(d, computed)
                            rows.append(dict(
                                base, arm=arm, arm_layers=";".join(layers),
                                n_arm=len(layers),
                                arm_hits=len(set(layers) & set(computed)),
                                acc=round(accuracy(m, Xev, yev, is4d), 4),
                                d_reached_after=round(dr, 4),
                                d_untouched_after=round(du, 4),
                                delta_all=json.dumps({k: round(v, 4) for k, v in d.items()})))
                    pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                              header=not os.path.exists(OUT))

    # -------------------------------------------------------- summary
    df = pd.read_csv(OUT)
    df = df[df.reachable == True]
    from scipy.stats import wilcoxon
    print("\n=== 表validate regenerated (2b) ===")
    for (m, R), g in df.groupby(["model", "budget"]):
        cells = []
        for arm in ARMS:
            ga = g[g.arm == arm]
            if len(ga):
                cells.append(f"{arm} {ga.acc.mean():.3f}")
        n_bn, n_rec = g.n_bn.iloc[0], g.n_recal.iloc[0]
        print(f"  {m:8s} {R}x  ({n_rec}/{n_bn})  " + "  ".join(cells))
    print("\n=== discriminating control: computed vs same-size wrong sets ===")
    for (m, R), g in df.groupby(["model", "budget"]):
        piv = g.pivot_table(index="subject", columns="arm", values="acc")
        if "computed" not in piv or g.n_bn.iloc[0] == 1:
            print(f"  {m} {R}x: single BN layer -- all arms identical by construction")
            continue
        for other in ["first_only", "random_k"]:
            d = piv["computed"] - piv[other]
            if d.std() > 0:
                print(f"  {m:8s} {R}x  computed - {other:10s} = {d.mean():+.3f} "
                      f"({int((d>0).sum())}/{len(d)}, p={wilcoxon(d).pvalue:.3f})")
        hits = g[g.arm == "random_k"].arm_hits.mean() / max(g.n_recal.iloc[0], 1)
        print(f"           random_k overlapped the computed set {hits:.0%} of the time")
    print("\n=== Δ at the reached layers after each arm (mechanism) ===")
    for (m, R), g in df.groupby(["model", "budget"]):
        cells = [f"{arm} {g[g.arm==arm].d_reached_after.mean():.3f}" for arm in ARMS
                 if len(g[g.arm == arm])]
        print(f"  {m:8s} {R}x  " + "  ".join(cells))


if __name__ == "__main__":
    main()
