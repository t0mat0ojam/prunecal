"""EXP-02 -- 枝刈り対象層の選択, canonical rerun (regenerates 表2).

Purpose: for every dense convolution in EEGNet / ShallowConvNet / Deep4Net,
solve Eq. 2 at matched MAC budgets (1.5x, 2.0x, 3.0x), physically prune with
the corrected semantics, and measure held-out accuracy before and after
repairing the probe-computed BatchNorm set on TRAINING data. Settles the
Deep4 conv_2-vs-conv_3 target question and re-verifies the ShallowConvNet
conv_spat negative control (empty repair set, collapse) on clean code.

Design deltas vs the original which_layer run (disclose in the paper):
  * physical pruning via prunecal (the parametrization silent-no-op bug that
    invalidated the original conv_spat arm cannot occur: pruning rewrites
    tensors, and the package's masked-equivalence test guards the surgery)
  * budget->sparsity via Eq. 2 on actually-pruned copies (cascading MACs
    counted identically for every layer)
  * criterion fixed to discriminability (EXP-01's pipeline default)
  * per-layer repair on training data logged alongside raw accuracy
  * split identical to EXP-01: 2b = LOSO over the three feedback sessions;
    F1=32, D=2, F2=64; 9 subjects x 3 seeds; no fine-tuning

Scope: 2b (development set). The chosen targets go to both datasets in the
multiarch rerun (EXP-03).

Run:  pip install moabb braindecode git+https://github.com/USER/prunecal.git
      MODELS=eegnet  python layer_rerun.py     # ~1.5 h on a T4
      MODELS=shallow python layer_rerun.py     # ~1 h
      MODELS=deep4   python layer_rerun.py     # ~3 h
      MOCK=1 SUBJECTS=1 SEEDS=1 EPOCHS=2 python layer_rerun.py   # smoke
Resumable per (dataset, model, subject, seed, fold); same CSV rules as
EXP-01: smoke output to a separate file, re-upload + exact filename to resume.
"""

import os
import types
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from prunecal import BudgetUnreachableError, compress, count_macs, recalibrate
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

DATASETS  = os.environ.get("DATASETS", "2b").split(",")
MODELS    = os.environ.get("MODELS", "eegnet,shallow,deep4").split(",")
SUBJECTS  = [int(s) for s in os.environ.get("SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
SEEDS     = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS_2B = int(os.environ.get("EPOCHS", "60"))
EPOCHS_2A = int(os.environ.get("EPOCHS_2A", os.environ.get("EPOCHS", "200")))
BUDGETS   = [1.5, 2.0, 3.0]
FEEDBACK_SESSIONS_2B = ["2train", "3test", "4test"]
OUT       = os.environ.get("OUT", "layer_rerun.csv")
DEV       = "cuda" if torch.cuda.is_available() else "cpu"


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
            for s in dict.fromkeys(sess)}


def load_mock(dataset, subject):
    rng = np.random.default_rng(subject)
    n_cls, C = (2, 3) if dataset == "2b" else (4, 22)
    names = [f"s{i}" for i in range(3 if dataset == "2b" else 2)]
    t = np.arange(576) / 128.0
    out = {}
    for s in names:
        y = (np.arange(96) % n_cls).astype("int64")
        X = rng.normal(0, 1, (96, C, 576)).astype("float32")
        for c in range(n_cls):
            X[y == c, c % C] += 1.3 * np.sin(2 * np.pi * (9 + 2 * c) * t)
        out[s] = (X, y)
    return out


def standardize(x):
    mu = x.mean(axis=(0, 2), keepdims=True)
    sd = x.std(axis=(0, 2), keepdims=True)
    return (x - mu) / (sd + 1e-8)


# ------------------------------------------------------------------- models
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


def candidate_layers(model):
    """Every dense (groups=1) convolution except the classifier head."""
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and m.groups == 1
            and "classifier" not in n]


def shape(is4d, x):
    return x.unsqueeze(1) if is4d else x


# ------------------------------------------------------------------- train/eval
def train(model, X, y, epochs, seed, is4d):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = shape(is4d, torch.tensor(X[idx])).to(DEV)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xb), torch.tensor(y[idx]).to(DEV)).backward()
            opt.step()
    return model.cpu().eval()


@torch.no_grad()
def accuracy(model, X, y, is4d):
    model.eval()
    hits = 0
    for i in range(0, len(X), 128):
        xb = shape(is4d, torch.tensor(X[i:i + 128]))
        hits += (model(xb).argmax(1).numpy() == y[i:i + 128]).sum()
    return hits / len(y)


# ------------------------------------------------------------------- main
def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.dataset, r.model, r.subject, r.seed, r.fold) for r in prev.itertuples()}
        print(f"resuming -- {len(done)} cells done")

    for dataset in DATASETS:
        epochs = EPOCHS_2B if dataset == "2b" else EPOCHS_2A
        for mname in MODELS:
            for subject in SUBJECTS:
                data = {s: (standardize(X), y) for s, (X, y) in
                        load_subject(dataset, subject).items()}
                sess = list(data.keys())
                for seed in SEEDS:
                    for held_out in sess:
                        if (dataset, mname, subject, seed, held_out) in done:
                            continue
                        tr = [s for s in sess if s != held_out]
                        Xtr = np.concatenate([data[s][0] for s in tr])
                        ytr = np.concatenate([data[s][1] for s in tr])
                        Xte, yte = data[held_out]
                        n_cls = int(ytr.max()) + 1

                        model, is4d = build_model(mname, Xtr.shape[1], Xtr.shape[2], n_cls)
                        dense = train(model, Xtr, ytr, epochs, seed, is4d)
                        acc_dense = accuracy(dense, Xte, yte, is4d)
                        x1 = shape(is4d, torch.tensor(Xtr[:1]))
                        total, per = count_macs(dense, x1)
                        x4tr = shape(is4d, torch.tensor(Xtr))
                        ytr_t = torch.tensor(ytr)
                        print(f"[{dataset} {mname} s{subject} seed{seed} fold {held_out}] "
                              f"dense={acc_dense:.3f}", flush=True)

                        rows = []
                        for layer in candidate_layers(dense):
                            share = per.get(layer, 0.0) / total
                            for R in BUDGETS:
                                base = dict(dataset=dataset, model=mname,
                                            subject=subject, seed=seed, fold=held_out,
                                            layer=layer, mac_share=round(share, 4),
                                            budget=R, dense=round(acc_dense, 4))
                                try:
                                    pruned, rep = compress(
                                        dense, x4tr, ytr_t, layer=layer, budget=R,
                                        criterion="discriminability")
                                except BudgetUnreachableError:
                                    rows.append(dict(base, reachable=False, sparsity=np.nan,
                                                     achieved=np.nan, n_recal=np.nan,
                                                     recal_layers="", acc_raw=np.nan,
                                                     acc=np.nan))
                                    continue
                                acc_raw = accuracy(pruned, Xte, yte, is4d)
                                if rep.recalibrate:
                                    recalibrate(pruned, x4tr, layers=rep.recalibrate)
                                rows.append(dict(
                                    base, reachable=True,
                                    sparsity=round(len(rep.pruned) /
                                                   (len(rep.pruned) + len(rep.kept)), 3),
                                    achieved=round(rep.macs_before / rep.macs_after, 3),
                                    n_recal=len(rep.recalibrate),
                                    recal_layers=";".join(rep.recalibrate),
                                    acc_raw=round(acc_raw, 4),
                                    acc=round(accuracy(pruned, Xte, yte, is4d), 4)))
                        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                                  header=not os.path.exists(OUT))

    # ----------------------------------------------------------- summary
    df = pd.read_csv(OUT)
    from scipy.stats import wilcoxon
    print("\n=== 表2 regenerated: mean accuracy (repaired) at each budget ===")
    for (m, layer), g in df.groupby(["model", "layer"]):
        share = g.mac_share.iloc[0]
        cells = []
        for R in BUDGETS:
            gr = g[g.budget == R]
            if gr.reachable.mean() < 0.5:
                cells.append(f"{R}x:到達不能")
            else:
                cells.append(f"{R}x:{gr.acc.mean():.3f}(raw {gr.acc_raw.mean():.3f}, "
                             f"n_recal {gr.n_recal.mean():.1f})")
        print(f"  {m:8s} {layer:28s} share {share:5.1%}  " + "  ".join(cells))
    print("\n=== target questions ===")
    for m, la, lb in [("deep4", "conv_2", "conv_3"),
                      ("shallow", "conv_time_spat.conv_time", "conv_time_spat.conv_spat")]:
        d = df[df.model == m]
        for R in BUDGETS:
            pa = d[(d.layer == la) & (d.budget == R)].groupby("subject").acc.mean()
            pb = d[(d.layer == lb) & (d.budget == R)].groupby("subject").acc.mean()
            both = pa.index.intersection(pb.index)
            if len(both) >= 5 and not (pa[both].isna().any() or pb[both].isna().any()):
                diff = pb[both] - pa[both]
                print(f"  {m} {lb} - {la} @ {R}x: {diff.mean():+.3f} "
                      f"({int((diff > 0).sum())}/{len(both)}, "
                      f"p={wilcoxon(diff).pvalue:.3f})")


if __name__ == "__main__":
    main()
