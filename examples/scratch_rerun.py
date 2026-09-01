"""EXP-05B -- 圧縮の是非：枝刈り＋修復 対 同規模モデルのゼロからの学習.

Answers the standard objection to structured pruning (Liu et al., ICLR
2019, "Rethinking the Value of Network Pruning"): why prune a large
trained model down to 4 filters instead of just training a 4-filter
model from scratch?

The comparison here is unusually clean, because pruning EEGNet(F1=32)
by 87.5% yields a model that is STRUCTURALLY IDENTICAL to a scratch
EEGNet(F1=4): same parameter shapes, same 552,088 MACs, same 3,378
parameters (verified in-script). The two arms therefore differ in
exactly one respect -- training history.

Arms:
  pruned_none    prune the trained large model, no adaptation
  pruned_repair  ... + probe-set recalibration on 16 unlabeled target trials
  scratch_none   train the same small architecture from scratch
  scratch_repair ... + the same recalibration

The mechanism makes a falsifiable prediction: the scratch model's
BatchNorm statistics are CORRECT (nothing was removed), so recalibration
should gain ~0 for it -- exactly as the dense model did in EXP-03
(-0.006) -- while the pruned model gains ~+0.13. If scratch_repair also
gains a lot, the "self-inflicted damage" account is wrong.

Three possible outcomes, all reportable:
  scratch ~= pruned+repair  -> the contribution is the mechanism, not a
                               performance win; say so plainly
  pruned+repair > scratch   -> pruning preserves something scratch
                               training does not find
  scratch > pruned+repair   -> honest scope limit: the pipeline is for
                               when you must compress an ALREADY-TRAINED
                               model (the deployment case), and we say so

Scope: 2b, EEGNet, ratio 0.875. Env vars extend it (RATIOS=0.5,0.75,0.875
triples the cost; DATASETS=2a adds the confirmation set).

Run:  MOCK=1 SUBJECTS=1 SEEDS=1 EPOCHS=2 OUT=scratch_SMOKE.csv python scratch_rerun.py
      OUT=scratch_rerun.csv python scratch_rerun.py            (~3.5 h)
"""

import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch

from prunecal import compress, count_macs, recalibrate
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

DATASETS = os.environ.get("DATASETS", "2b").split(",")
RATIOS   = [float(r) for r in os.environ.get("RATIOS", "0.875").split(",")]
SUBJECTS = [int(s) for s in os.environ.get("SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
SEEDS    = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS   = int(os.environ.get("EPOCHS", "60"))
EPOCHS_2A = int(os.environ.get("EPOCHS_2A", "200"))
F1_BASE, D, F2 = 32, 2, 64
N_CALIB  = 16
FEEDBACK_SESSIONS_2B = ["2train", "3test", "4test"]
OUT      = os.environ.get("OUT", "scratch_rerun.csv")
DEV      = "cuda" if torch.cuda.is_available() else "cpu"


def load_subject(dataset, subject):
    if os.environ.get("MOCK", "0") == "1":
        rng = np.random.default_rng(subject)
        n_cls, C = (2, 3) if dataset == "2b" else (4, 22)
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
    lut = {n: i for i, n in enumerate(sorted(set(np.asarray(y))))}
    y = np.array([lut[v] for v in np.asarray(y)], dtype="int64")
    T = (X.shape[-1] // 32) * 32
    return {s: (X[sess==s,:,:T].astype("float32"), y[sess==s]) for s in dict.fromkeys(sess)}


def standardize(x):
    mu = x.mean(axis=(0,2), keepdims=True); sd = x.std(axis=(0,2), keepdims=True)
    return (x-mu)/(sd+1e-8)


def train(model, X, y, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(
                model(torch.tensor(X[idx]).unsqueeze(1).to(DEV)),
                torch.tensor(y[idx]).to(DEV)).backward()
            opt.step()
    return model.cpu().eval()


@torch.no_grad()
def accuracy(model, X, y):
    model.eval(); hits = 0
    for i in range(0, len(X), 128):
        hits += (model(torch.tensor(X[i:i+128]).unsqueeze(1)).argmax(1).numpy()
                 == y[i:i+128]).sum()
    return hits/len(y)


def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.dataset, r.subject, r.seed, r.fold, r.ratio) for r in prev.itertuples()}
        print(f"resuming -- {len(done)} cells done")

    for dataset in DATASETS:
        epochs = EPOCHS if dataset == "2b" else EPOCHS_2A
        for subject in SUBJECTS:
            data = {s: (standardize(X), y) for s,(X,y) in load_subject(dataset, subject).items()}
            sess = list(data.keys())
            for seed in SEEDS:
                for held_out in sess:
                    tr = [s for s in sess if s != held_out]
                    Xtr = np.concatenate([data[s][0] for s in tr])
                    ytr = np.concatenate([data[s][1] for s in tr])
                    Xtg, ytg = data[held_out]
                    Xcal = Xtg[:N_CALIB]
                    Xev, yev = Xtg[N_CALIB:], ytg[N_CALIB:]
                    n_cls = int(ytr.max())+1
                    C, T = Xtr.shape[1], Xtr.shape[2]
                    x4tr = torch.tensor(Xtr).unsqueeze(1)
                    x4cal = torch.tensor(Xcal).unsqueeze(1)

                    todo = [r for r in RATIOS
                            if (dataset, subject, seed, held_out, r) not in done]
                    if not todo:
                        continue
                    dense = train(EEGNet(C, T, n_cls, F1=F1_BASE, D=D, F2=F2),
                                  Xtr, ytr, epochs, seed)
                    acc_dense = accuracy(dense, Xev, yev)
                    macs_dense, _ = count_macs(dense, x4tr[:1])
                    print(f"[{dataset} s{subject} seed{seed} {held_out}] dense={acc_dense:.3f}",
                          flush=True)

                    rows = []
                    for ratio in todo:
                        # --- prune the trained large model ---
                        pruned, rep = compress(dense, x4tr, torch.tensor(ytr),
                                               layer="conv_temporal", ratio=ratio,
                                               criterion="discriminability")
                        macs_p, _ = count_macs(pruned, x4tr[:1])
                        acc_pn = accuracy(pruned, Xev, yev)
                        m = copy.deepcopy(pruned)
                        if rep.recalibrate:
                            recalibrate(m, x4cal, layers=rep.recalibrate)
                        acc_pr = accuracy(m, Xev, yev)

                        # --- same architecture, trained from scratch ---
                        f1_small = max(int(round(F1_BASE*(1-ratio))), 1)
                        scratch = train(EEGNet(C, T, n_cls, F1=f1_small, D=D, F2=F2),
                                        Xtr, ytr, epochs, seed+1000)
                        macs_s, _ = count_macs(scratch, x4tr[:1])
                        acc_sn = accuracy(scratch, Xev, yev)
                        s2 = copy.deepcopy(scratch)
                        if rep.recalibrate:
                            recalibrate(s2, x4cal, layers=rep.recalibrate)
                        acc_sr = accuracy(s2, Xev, yev)

                        rows.append(dict(
                            dataset=dataset, subject=subject, seed=seed, fold=held_out,
                            ratio=ratio, f1_small=f1_small,
                            macs_dense=macs_dense, macs_pruned=macs_p, macs_scratch=macs_s,
                            macs_match=int(macs_p == macs_s),
                            params_pruned=sum(p.numel() for p in pruned.parameters()),
                            params_scratch=sum(p.numel() for p in scratch.parameters()),
                            recal_layers=";".join(rep.recalibrate),
                            acc_dense=round(acc_dense,4),
                            pruned_none=round(acc_pn,4), pruned_repair=round(acc_pr,4),
                            scratch_none=round(acc_sn,4), scratch_repair=round(acc_sr,4)))
                    pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                              header=not os.path.exists(OUT))

    # ---------------------------------------------------------- summary
    df = pd.read_csv(OUT)
    from scipy.stats import wilcoxon
    for (ds, r), g in df.groupby(["dataset","ratio"]):
        print(f"\n=== {ds} / ratio {r} (F1 {F1_BASE} -> {g.f1_small.iloc[0]}, "
              f"MACs match: {bool(g.macs_match.all())}) ===")
        print(f"  dense              {g.acc_dense.mean():.3f}")
        for c in ["pruned_none","pruned_repair","scratch_none","scratch_repair"]:
            print(f"  {c:18s} {g[c].mean():.3f}")
        per = g.groupby("subject")[["pruned_none","pruned_repair",
                                    "scratch_none","scratch_repair"]].mean()
        def cmp(a, b):
            d = per[a]-per[b]
            return (f"{d.mean():+.3f} ({int((d>0).sum())}/{len(d)}, "
                    f"p={wilcoxon(d).pvalue:.3f})") if d.std()>0 else "identical"
        print(f"\n  KEY: pruned_repair − scratch_none   {cmp('pruned_repair','scratch_none')}")
        print(f"       pruned_repair − scratch_repair {cmp('pruned_repair','scratch_repair')}")
        print(f"\n  MECHANISM CHECK (repair gain; prediction: large for pruned, ~0 for scratch)")
        print(f"       pruned:  {cmp('pruned_repair','pruned_none')}")
        print(f"       scratch: {cmp('scratch_repair','scratch_none')}")


if __name__ == "__main__":
    main()
