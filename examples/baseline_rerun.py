"""EXP-05A -- 既存の較正不要適応手法との比較 (regenerates 表pareto).

Compression is held FIXED (87.5% pruning of the EXP-02 target layer, the
discriminability criterion) and only the adaptation varies. Every arm
sees exactly 16 target-session trials; only `oracle16` sees their labels.

Arms and what each costs on a device:
  none        --                              no adaptation
  ea          input transform                 Euclidean Alignment. Requires
              its own model TRAINED ON ALIGNED DATA (feeding aligned data to
              an unaligned model is not a valid control -- see the report's
              method section). Reference covariance is estimated from the
              same 16 trials, so the adaptation budget is matched.
  tent_*      backprop                        TENT, swept over lr x steps.
              GENEROUS implementation: BatchNorm running statistics update
              during adaptation AND the affine parameters are optimised by
              entropy minimisation, so TENT receives everything AdaBN gets
              plus the gradient step. Report the best configuration.
  computed    --                              probe-set recalibration (ours)
  all         --                              full AdaBN
  oracle16    LABELS + backprop               supervised fine-tuning on the
              16 trials WITH labels. Not calibration-free; included to price
              what label-freeness costs. Swept over lr x epochs.

Scope: 2b (development set), EEGNet, ratio 0.875 -- the report's 表pareto
scope. MODELS/DATASETS/RATIOS are env vars if you want to extend it.

Run:  OUT=baseline_rerun.csv python baseline_rerun.py            (~3.5 h)
      MOCK=1 SUBJECTS=1 SEEDS=1 EPOCHS=2 OUT=base_SMOKE.csv python baseline_rerun.py
Note: trains TWO models per cell (raw + EA-aligned), so it is ~2x the cost
of a single-model experiment.
"""

import copy
import os
import types
import warnings

import numpy as np
import pandas as pd
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from prunecal import compress, recalibrate
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

DATASETS = os.environ.get("DATASETS", "2b").split(",")
MODELS   = os.environ.get("MODELS", "eegnet").split(",")
RATIOS   = [float(r) for r in os.environ.get("RATIOS", "0.875").split(",")]
SUBJECTS = [int(s) for s in os.environ.get("SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
SEEDS    = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS   = int(os.environ.get("EPOCHS", "60"))
EPOCHS_2A = int(os.environ.get("EPOCHS_2A", "200"))
N_CALIB  = 16
TENT_GRID   = [(1e-4, 1), (1e-4, 10), (1e-3, 1), (1e-3, 10), (1e-2, 1), (1e-2, 10)]
ORACLE_GRID = [(1e-4, 10), (1e-4, 30), (1e-3, 10), (1e-3, 30)]
FEEDBACK_SESSIONS_2B = ["2train", "3test", "4test"]
OUT      = os.environ.get("OUT", "baseline_rerun.csv")
DEV      = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS  = {("2b","eegnet"):"conv_temporal", ("2a","eegnet"):"conv_temporal",
            ("2b","shallow"):"conv_time_spat.conv_time",
            ("2a","shallow"):"conv_time_spat.conv_time",
            ("2b","deep4"):"conv_3", ("2a","deep4"):"conv_time_spat.conv_spat"}
COST = {  # (labels, backprop, input transform)
    "none": (0,0,0), "ea": (0,0,1), "tent": (0,1,0),
    "computed": (0,0,0), "all": (0,0,0), "oracle16": (1,1,0)}


# ------------------------------------------------------------------ data
def load_subject(dataset, subject):
    if os.environ.get("MOCK","0") == "1":
        rng = np.random.default_rng(subject)
        n_cls, C = (2,3) if dataset=="2b" else (4,22)
        names = ["2train","3test","4test"] if dataset=="2b" else ["0train","1test"]
        t = np.arange(576)/128.0
        out = {}
        for si, s in enumerate(names):
            y = (np.arange(128) % n_cls).astype("int64")
            X = rng.normal(0,1,(128,C,576)).astype("float32")
            for c in range(n_cls):
                X[y==c, c%C] += 1.3*np.sin(2*np.pi*(9+2*c)*t)
            out[s] = (X*(1+0.2*si)+0.1*si, y)
        return out
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
    lut = {n:i for i,n in enumerate(sorted(set(np.asarray(y))))}
    y = np.array([lut[v] for v in np.asarray(y)], dtype="int64")
    T = (X.shape[-1]//32)*32
    return {s: (X[sess==s,:,:T].astype("float32"), y[sess==s]) for s in dict.fromkeys(sess)}


def standardize(x):
    mu = x.mean(axis=(0,2), keepdims=True); sd = x.std(axis=(0,2), keepdims=True)
    return (x-mu)/(sd+1e-8)


def ea_matrix(x):
    """Euclidean Alignment: R^{-1/2} from the mean spatial covariance."""
    R = np.einsum("nct,ndt->cd", x, x) / (x.shape[0]*x.shape[2])
    w, V = np.linalg.eigh(R)
    return (V @ np.diag(1.0/np.sqrt(np.maximum(w, 1e-10))) @ V.T).astype("float32")


def ea_apply(x, M):
    return np.einsum("cd,ndt->nct", M, x).astype("float32")


# ------------------------------------------------------------------ models
def _unfuse(m):
    for mod in m.modules():
        if type(mod).__name__ == "CombinedConv":
            mod.forward = types.MethodType(lambda s,x: s.conv_spat(s.conv_time(x)), mod)
    return m


def build_model(name, C, T, n_cls):
    if name == "eegnet":
        return EEGNet(C, T, n_cls, F1=32, D=2, F2=64), True
    from braindecode.models import Deep4Net, ShallowFBCSPNet
    return _unfuse({"shallow":ShallowFBCSPNet,"deep4":Deep4Net}[name](
        n_chans=C, n_outputs=n_cls, n_times=T)), False


def shape(is4d, x):
    return x.unsqueeze(1) if is4d else x


def train(model, X, y, epochs, seed, is4d):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(
                model(shape(is4d, torch.tensor(X[idx])).to(DEV)),
                torch.tensor(y[idx]).to(DEV)).backward()
            opt.step()
    return model.cpu().eval()


@torch.no_grad()
def accuracy(model, X, y, is4d):
    model.eval(); hits = 0
    for i in range(0, len(X), 128):
        hits += (model(shape(is4d, torch.tensor(X[i:i+128]))).argmax(1).numpy()
                 == y[i:i+128]).sum()
    return hits/len(y)


# ------------------------------------------------------------------ arms
def tent_adapt(model, x, lr, steps):
    """TENT, generous variant: BN running stats update during adaptation
    (so TENT also gets the AdaBN effect) and the affine parameters are
    optimised by entropy minimisation."""
    m = copy.deepcopy(model).to(DEV)
    m.eval()
    params = []
    for mod in m.modules():
        if isinstance(mod, _BatchNorm):
            mod.train()                       # batch stats + running-stat update
            if mod.weight is not None:
                params += [mod.weight, mod.bias]
    if not params:
        return m.cpu().eval()
    opt = torch.optim.Adam(params, lr=lr)
    xb = x.to(DEV)
    for _ in range(steps):
        p = torch.softmax(m(xb), dim=1)
        loss = -(p*torch.log(p+1e-8)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.cpu().eval()


def oracle_finetune(model, x, y, lr, epochs):
    m = copy.deepcopy(model).to(DEV).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xb, yb = x.to(DEV), y.to(DEV)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(xb), yb).backward()
        opt.step()
    return m.cpu().eval()


# ------------------------------------------------------------------ main
def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.dataset, r.model, r.subject, r.seed, r.fold, r.ratio)
                for r in prev.itertuples()}
        print(f"resuming -- {len(done)} cells done")

    for dataset in DATASETS:
        epochs = EPOCHS if dataset == "2b" else EPOCHS_2A
        for mname in MODELS:
            target = TARGETS[(dataset, mname)]
            for subject in SUBJECTS:
                raw = {s: (standardize(X), y) for s,(X,y) in
                       load_subject(dataset, subject).items()}
                sess = list(raw.keys())
                for seed in SEEDS:
                    for held_out in sess:
                        for ratio in RATIOS:
                            if (dataset,mname,subject,seed,held_out,ratio) in done:
                                continue
                            tr = [s for s in sess if s != held_out]
                            Xtr = np.concatenate([raw[s][0] for s in tr])
                            ytr = np.concatenate([raw[s][1] for s in tr])
                            Xtg, ytg = raw[held_out]
                            Xcal, ycal = Xtg[:N_CALIB], ytg[:N_CALIB]
                            Xev, yev = Xtg[N_CALIB:], ytg[N_CALIB:]
                            n_cls = int(ytr.max())+1

                            # ---- raw pipeline ----
                            model, is4d = build_model(mname, Xtr.shape[1], Xtr.shape[2], n_cls)
                            dense = train(model, Xtr, ytr, epochs, seed, is4d)
                            x4tr = shape(is4d, torch.tensor(Xtr))
                            pruned, rep = compress(dense, x4tr, torch.tensor(ytr),
                                                   layer=target, ratio=ratio,
                                                   criterion="discriminability")
                            computed = list(rep.recalibrate)
                            x4cal = shape(is4d, torch.tensor(Xcal))
                            acc_dense = accuracy(dense, Xev, yev, is4d)

                            # ---- EA pipeline: align per session, retrain ----
                            Xtr_ea = np.concatenate([ea_apply(raw[s][0], ea_matrix(raw[s][0]))
                                                     for s in tr])
                            M_tg = ea_matrix(Xcal)          # matched 16-trial budget
                            Xcal_ea, Xev_ea = ea_apply(Xcal, M_tg), ea_apply(Xev, M_tg)
                            m_ea, _ = build_model(mname, Xtr.shape[1], Xtr.shape[2], n_cls)
                            dense_ea = train(m_ea, Xtr_ea, ytr, epochs, seed, is4d)
                            pruned_ea, _ = compress(dense_ea,
                                                    shape(is4d, torch.tensor(Xtr_ea)),
                                                    torch.tensor(ytr), layer=target,
                                                    ratio=ratio, criterion="discriminability")
                            print(f"[{dataset} {mname} s{subject} seed{seed} {held_out} "
                                  f"r{ratio}] dense={acc_dense:.3f} probe={computed}", flush=True)

                            rows = []
                            def add(arm, cfg, acc, n_recal=0):
                                lab, bp, tf = COST[arm.split("_")[0] if arm.startswith(("tent","oracle")) else arm]
                                rows.append(dict(dataset=dataset, model=mname, subject=subject,
                                                 seed=seed, fold=held_out, ratio=ratio,
                                                 layer=target, arm=arm, config=cfg,
                                                 acc=round(acc,4), n_recal=n_recal,
                                                 n_bn=len(rep.probe.reached)+len(rep.probe.untouched),
                                                 needs_labels=lab, needs_backprop=bp,
                                                 needs_input_transform=tf,
                                                 acc_dense=round(acc_dense,4)))

                            add("none", "", accuracy(pruned, Xev, yev, is4d))
                            add("ea", "16-trial ref", accuracy(pruned_ea, Xev_ea, yev, is4d))
                            m = copy.deepcopy(pruned)
                            if computed: recalibrate(m, x4cal, layers=computed)
                            add("computed", "", accuracy(m, Xev, yev, is4d), len(computed))
                            m = copy.deepcopy(pruned); recalibrate(m, x4cal)
                            add("all", "", accuracy(m, Xev, yev, is4d), len(rep.probe.reached)+len(rep.probe.untouched))
                            for lr, st in TENT_GRID:
                                add(f"tent", f"lr{lr:g}_s{st}",
                                    accuracy(tent_adapt(pruned, x4cal, lr, st), Xev, yev, is4d))
                            for lr, ep in ORACLE_GRID:
                                add(f"oracle16", f"lr{lr:g}_e{ep}",
                                    accuracy(oracle_finetune(pruned, x4cal,
                                             torch.tensor(ycal), lr, ep), Xev, yev, is4d))
                            # EA combined with probe-set recalibration (orthogonal)
                            m = copy.deepcopy(pruned_ea)
                            if computed: recalibrate(m, shape(is4d, torch.tensor(Xcal_ea)), layers=computed)
                            add("ea", "+probe-set", accuracy(m, Xev_ea, yev, is4d), len(computed))
                            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                                      header=not os.path.exists(OUT))

    # -------------------------------------------------------- summary
    df = pd.read_csv(OUT)
    from scipy.stats import wilcoxon
    for (ds, m, r), g in df.groupby(["dataset","model","ratio"]):
        print(f"\n=== {ds} / {m} / ratio {r}  (dense {g.acc_dense.mean():.3f}) ===")
        best = {}
        for arm, ga in g.groupby("arm"):
            if arm in ("tent","oracle16"):
                per_cfg = ga.groupby("config").acc.mean()
                cfg = per_cfg.idxmax(); best[arm] = (cfg, per_cfg.max())
            elif arm == "ea":
                for cfg, gc in ga.groupby("config"):
                    best[f"ea {cfg}"] = (cfg, gc.acc.mean())
            else:
                best[arm] = ("", ga.acc.mean())
        ours = g[(g.arm=="computed")].groupby("subject").acc.mean()
        print(f"{'arm':22s} {'config':14s} {'acc':>7s} {'vs ours':>9s} {'lab':>4s} {'bp':>3s} {'tf':>3s} {'層':>3s}")
        for k,(cfg,v) in sorted(best.items(), key=lambda kv: kv[1][1]):
            sub = g[(g.arm==k.split(" ")[0]) & ((g.config==cfg) if cfg else True)]
            other = sub.groupby("subject").acc.mean()
            idx = ours.index.intersection(other.index)
            d = ours[idx]-other[idx]
            p = f"{wilcoxon(d).pvalue:.3f}" if d.std()>0 else "  --"
            c = sub.iloc[0]
            print(f"{k:22s} {cfg:14s} {v:>7.3f} {d.mean():>+7.3f} {p:>6s} "
                  f"{c.needs_labels:>3d} {c.needs_backprop:>3d} {c.needs_input_transform:>3d} "
                  f"{c.n_recal:>3d}")


if __name__ == "__main__":
    main()
