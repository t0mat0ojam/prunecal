"""EXP-03 -- 圧縮率と較正不要適応の関係 (regenerates §4: frontier, Δ, separation).

Measures three interlocking things at each pruning ratio, for every
(dataset x architecture x subject x seed x fold):

  1. FRONTIER   accuracy with no adaptation / probe-set recalibration /
                full AdaBN, where recalibration uses 16 UNLABELED trials
                from the TARGET session (labels never touched).
                Headline: the adaptation gain grows with the pruning ratio.
  2. Δ TRAJECTORY  Eq. 3 per BatchNorm layer, before and after adaptation,
                split into probe-set ("reached") and untouched layers.
                Headline: only reached layers' Δ grows with pruning.
  3. SEPARATION  the same Δ measured on held-out data from the TRAINING
                sessions, where no session change exists. Whatever Δ
                appears there is pruning-induced; the difference from the
                target-session Δ is the session component.

Targets are EXP-02's confirmed choices (data-free where they differ):
  eegnet  -> conv_temporal              (both datasets)
  shallow -> conv_time_spat.conv_time   (both datasets)
  deep4   -> conv_3 on 2b; conv_time_spat.conv_spat on 2a
             (2a: conv_2/3/4 are all 到達不能 because after the spatial
              convolution the electrode dimension is gone, so the later
              blocks' cost is electrode-independent; see EXP-02 log)

Protocol matches EXP-01/02: F1=32,D=2,F2=64; no fine-tuning; physical
pruning; discriminability criterion; per-session standardization;
2b = LOSO over the three feedback sessions, 2a = LOSO over its two.
Training sessions are split 80/20; the 20% is held out for the
separation measurement only (never trained on, never adapted on).

Run (chunk by dataset x model -- 6 chunks):
  !DATASETS=2b MODELS=eegnet  OUT=frontier_rerun.csv python frontier_rerun.py
  ...
  MOCK=1 SUBJECTS=1 SEEDS=1 EPOCHS=2 python frontier_rerun.py   # smoke
Resumable per (dataset, model, subject, seed, fold); same CSV rules.
"""

import json
import os
import types
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from prunecal import compress, delta, probe, recalibrate
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

DATASETS  = os.environ.get("DATASETS", "2b,2a").split(",")
MODELS    = os.environ.get("MODELS", "eegnet,shallow,deep4").split(",")
SUBJECTS  = [int(s) for s in os.environ.get("SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
SEEDS     = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS_2B = int(os.environ.get("EPOCHS", "60"))
EPOCHS_2A = int(os.environ.get("EPOCHS_2A", os.environ.get("EPOCHS", "200")))
RATIOS    = [0.0, 0.2, 0.4, 0.6, 0.8, 0.875]
N_CALIB   = 16
SEP_FRAC  = 0.2
FEEDBACK_SESSIONS_2B = ["2train", "3test", "4test"]
OUT       = os.environ.get("OUT", "frontier_rerun.csv")
DEV       = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = {
    ("2b", "eegnet"):  "conv_temporal",
    ("2a", "eegnet"):  "conv_temporal",
    ("2b", "shallow"): "conv_time_spat.conv_time",
    ("2a", "shallow"): "conv_time_spat.conv_time",
    ("2b", "deep4"):   "conv_3",
    ("2a", "deep4"):   "conv_time_spat.conv_spat",
}


# ------------------------------------------------------------------ data
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
    for si, s in enumerate(names):
        y = (np.arange(128) % n_cls).astype("int64")
        X = rng.normal(0, 1, (128, C, 576)).astype("float32")
        for c in range(n_cls):
            X[y == c, c % C] += 1.3 * np.sin(2 * np.pi * (9 + 2 * c) * t)
        X = X * (1.0 + 0.25 * si) + 0.15 * si          # planted session shift
        out[s] = (X, y)
    return out


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
            torch.nn.functional.cross_entropy(
                model(xb), torch.tensor(y[idx]).to(DEV)).backward()
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


def split_delta(d, reached):
    """Mean Δ over probe-set layers and over the untouched ones."""
    r = [v for k, v in d.items() if k in reached]
    u = [v for k, v in d.items() if k not in reached]
    return (float(np.mean(r)) if r else np.nan,
            float(np.mean(u)) if u else np.nan)


# ------------------------------------------------------------------ main
def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.dataset, r.model, r.subject, r.seed, r.fold) for r in prev.itertuples()}
        print(f"resuming -- {len(done)} cells done")

    for dataset in DATASETS:
        epochs = EPOCHS_2B if dataset == "2b" else EPOCHS_2A
        for mname in MODELS:
            target = TARGETS[(dataset, mname)]
            for subject in SUBJECTS:
                data = {s: (standardize(X), y) for s, (X, y) in
                        load_subject(dataset, subject).items()}
                sess = list(data.keys())
                for seed in SEEDS:
                    for held_out in sess:
                        if (dataset, mname, subject, seed, held_out) in done:
                            continue
                        tr = [s for s in sess if s != held_out]
                        Xall = np.concatenate([data[s][0] for s in tr])
                        yall = np.concatenate([data[s][1] for s in tr])
                        rng = np.random.default_rng(seed)
                        perm = rng.permutation(len(Xall))
                        n_sep = max(int(SEP_FRAC * len(Xall)), 16)
                        sep_idx, tr_idx = perm[:n_sep], perm[n_sep:]
                        Xtr, ytr = Xall[tr_idx], yall[tr_idx]
                        Xsep = Xall[sep_idx]                    # separation set
                        Xtg, ytg = data[held_out]
                        Xcal = Xtg[:N_CALIB]                    # unlabeled
                        Xev, yev = Xtg[N_CALIB:], ytg[N_CALIB:]
                        n_cls = int(yall.max()) + 1

                        model, is4d = build_model(mname, Xtr.shape[1], Xtr.shape[2], n_cls)
                        dense = train(model, Xtr, ytr, epochs, seed, is4d)
                        x4tr, x4cal = shape(is4d, torch.tensor(Xtr)), shape(is4d, torch.tensor(Xcal))
                        x4sep = shape(is4d, torch.tensor(Xsep))
                        reached = probe(dense, target, x4tr[:1]).reached
                        acc_dense = accuracy(dense, Xev, yev, is4d)
                        print(f"[{dataset} {mname} s{subject} seed{seed} fold {held_out}] "
                              f"dense={acc_dense:.3f} probe-set={reached}", flush=True)

                        rows = []
                        for ratio in RATIOS:
                            if ratio == 0.0:
                                import copy
                                pruned, rec_layers = copy.deepcopy(dense), list(reached)
                                sparsity, achieved = 0.0, 1.0
                            else:
                                pruned, rep = compress(dense, x4tr, torch.tensor(ytr),
                                                       layer=target, ratio=ratio,
                                                       criterion="discriminability")
                                rec_layers = list(rep.recalibrate)
                                sparsity = len(rep.pruned) / (len(rep.pruned) + len(rep.kept))
                                achieved = rep.macs_before / rep.macs_after

                            # Δ before adaptation: target session vs training-held-out
                            d_tg = delta(pruned, x4cal)
                            d_sep = delta(pruned, x4sep)
                            tg_r, tg_u = split_delta(d_tg, rec_layers)
                            sp_r, sp_u = split_delta(d_sep, rec_layers)

                            acc_none = accuracy(pruned, Xev, yev, is4d)
                            import copy
                            m_probe, m_full = copy.deepcopy(pruned), copy.deepcopy(pruned)
                            if rec_layers:
                                recalibrate(m_probe, x4cal, layers=rec_layers)
                            recalibrate(m_full, x4cal)
                            d_after = delta(m_probe, x4cal)
                            af_r, af_u = split_delta(d_after, rec_layers)

                            rows.append(dict(
                                dataset=dataset, model=mname, subject=subject, seed=seed,
                                fold=held_out, layer=target, ratio=ratio,
                                sparsity=round(sparsity, 3), achieved=round(achieved, 3),
                                n_bn=len(d_tg), n_recal=len(rec_layers),
                                recal_layers=";".join(rec_layers),
                                acc_dense=round(acc_dense, 4),
                                acc_none=round(acc_none, 4),
                                acc_probe=round(accuracy(m_probe, Xev, yev, is4d), 4),
                                acc_full=round(accuracy(m_full, Xev, yev, is4d), 4),
                                d_target_reached=round(tg_r, 4), d_target_untouched=round(tg_u, 4),
                                d_sep_reached=round(sp_r, 4), d_sep_untouched=round(sp_u, 4),
                                d_after_reached=round(af_r, 4), d_after_untouched=round(af_u, 4),
                                delta_target=json.dumps({k: round(v, 4) for k, v in d_tg.items()}),
                                delta_sep=json.dumps({k: round(v, 4) for k, v in d_sep.items()}),
                                delta_after=json.dumps({k: round(v, 4) for k, v in d_after.items()}),
                                n_train=len(Xtr), n_sep=len(Xsep), n_eval=len(Xev)))
                        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                                  header=not os.path.exists(OUT))

    # -------------------------------------------------------- summary
    df = pd.read_csv(OUT)
    from scipy.stats import wilcoxon
    for (ds, m), g in df.groupby(["dataset", "model"]):
        print(f"\n=== {ds} / {m}  (target {g.layer.iloc[0]}, "
              f"repair {g.n_recal.iloc[-1]:.0f}/{g.n_bn.iloc[-1]:.0f}) ===")
        print(f"{'ratio':>6} {'dense':>7} {'none':>7} {'probe':>7} {'full':>7} "
              f"{'gain':>7} {'Δrch':>7} {'Δunt':>7} {'Δsep':>7} {'Δaft':>7}")
        for ratio, gr in g.groupby("ratio"):
            per = gr.groupby("subject")
            gain = (gr.acc_probe - gr.acc_none).mean()
            print(f"{ratio:>6.3f} {gr.acc_dense.mean():>7.3f} {gr.acc_none.mean():>7.3f} "
                  f"{gr.acc_probe.mean():>7.3f} {gr.acc_full.mean():>7.3f} "
                  f"{gain:>+7.3f} {gr.d_target_reached.mean():>7.3f} "
                  f"{gr.d_target_untouched.mean():>7.3f} {gr.d_sep_reached.mean():>7.3f} "
                  f"{gr.d_after_reached.mean():>7.3f}")
        lo = g[g.ratio == 0.0]; hi = g[g.ratio == max(RATIOS)]
        gl = (lo.acc_probe - lo.acc_none).groupby(lo.subject).mean()
        gh = (hi.acc_probe - hi.acc_none).groupby(hi.subject).mean()
        if len(gl) >= 5:
            d = gh - gl
            print(f"  gain at 0% = {gl.mean():+.3f} -> at {max(RATIOS):.1%} = {gh.mean():+.3f}"
                  f"  (increase {d.mean():+.3f}, {int((d>0).sum())}/{len(d)}, "
                  f"p={wilcoxon(d).pvalue:.3f})")
            dsep = hi.d_sep_reached.mean() - lo.d_sep_reached.mean()
            dses = (hi.d_target_reached.mean() - hi.d_sep_reached.mean())
            print(f"  separation: pruning-induced ΔΔ = {dsep:+.3f} | "
                  f"session component at max ratio = {dses:+.3f}")
            pf = (hi.acc_probe - hi.acc_full).groupby(hi.subject).mean()
            print(f"  probe-set vs full AdaBN at max ratio: {pf.mean():+.4f} "
                  f"(p={wilcoxon(pf).pvalue:.3f})" if pf.std() > 0 else
                  f"  probe-set vs full AdaBN: identical")


if __name__ == "__main__":
    main()
