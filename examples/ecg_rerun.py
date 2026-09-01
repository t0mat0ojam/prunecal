"""EXP-06 -- 一般化：脳波以外の生体信号での機構検証 (ECG, CinC 2017).

The wiring survey (probe_survey.py) already shows the probe applies to
non-EEG architectures. This experiment tests the MECHANISM itself in a
second signal domain, at the same standard as EXP-03: does pruning
inflate Δ at exactly the probe-named layers, and does a label-free
repair fix it?

Domain: PhysioNet/CinC 2017 single-lead ECG, normal rhythm vs atrial
fibrillation (balanced, chance = 0.5). Downloads openly, no registration.

Shift: CROSS-SUBJECT. Each recording is a different person, so splitting
by recording means the calibration and evaluation data come from people
the model never trained on -- the deployment scenario for a wearable
monitor, and the ECG analogue of the report's session shift.

Two architectures, chosen to be the design-guideline pair:
  eegnet    the report's own architecture on 1-lead ECG. Probe set 1/3.
  ecgresnet a Hannun et al. (2019)-style residual network.
            REIMPLEMENTED from the paper's description -- swap in the
            official Stanford code before citing any number from it.
            Probe set 32/33.
Same data, same treatment; the architecture alone decides whether repair
means one layer or nearly all of them.

Measured per (model, split, ratio), mirroring EXP-03:
  acc_none / acc_probe / acc_full   (repair uses 16 UNLABELED target-subject
                                     recordings; labels never touched)
  Δ at reached vs untouched layers, before and after repair
  separation: the same Δ measured on held-out TRAINING recordings, where
  no subject shift exists -- so whatever Δ appears there is pruning-induced

Run:  OUT=ecg_rerun.csv python ecg_rerun.py                 (~2 h)
      MOCK=1 SPLITS=2 EPOCHS=2 OUT=ecg_SMOKE.csv python ecg_rerun.py
"""

import copy
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from prunecal import compress, delta, probe, recalibrate
from prunecal.models import EEGNet

warnings.filterwarnings("ignore")

MODELS   = os.environ.get("MODELS", "eegnet,ecgresnet").split(",")
SPLITS   = int(os.environ.get("SPLITS", "9"))
EPOCHS   = int(os.environ.get("EPOCHS", "30"))
RATIOS   = [0.0, 0.2, 0.4, 0.6, 0.8, 0.875]
WINDOW   = 2048            # ~6.8 s at 300 Hz
N_CALIB  = 16
N_BLOCKS = int(os.environ.get("N_BLOCKS", "16"))
OUT      = os.environ.get("OUT", "ecg_rerun.csv")
DEV      = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS  = {"eegnet": "conv_temporal", "ecgresnet": "stem"}


# ------------------------------------------------------------------ model
class HannunECGResNet(nn.Module):
    """[REIMPLEMENTED] 1-D residual ECG network after Hannun et al.,
    Nature Medicine 2019. Written from the paper's description: k=15
    convolutions, 16 residual blocks, channels doubling every 4 blocks,
    subsampling every other block. Deviations to verify against the
    official code (github.com/awni/ecg): post-activation block order,
    strided 1x1 shortcut where the original may zero-pad, odd kernel."""

    def __init__(self, n_classes=2, base=32, n_blocks=16, k=15):
        super().__init__()
        self.stem = nn.Conv1d(1, base, k, padding=k//2, bias=False)
        self.stem_bn = nn.BatchNorm1d(base)
        blocks, shortcuts, ch = [], [], base
        for i in range(n_blocks):
            out = base * (2 ** (i//4)); stride = 2 if i % 2 else 1
            blocks.append(nn.Sequential(
                nn.Conv1d(ch, out, k, stride=stride, padding=k//2, bias=False),
                nn.BatchNorm1d(out), nn.ReLU(), nn.Dropout(0.2),
                nn.Conv1d(out, out, k, padding=k//2, bias=False), nn.BatchNorm1d(out)))
            shortcuts.append(nn.Identity() if (ch == out and stride == 1)
                             else nn.Conv1d(ch, out, 1, stride=stride, bias=False))
            ch = out
        self.blocks, self.shortcuts = nn.ModuleList(blocks), nn.ModuleList(shortcuts)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        for b, s in zip(self.blocks, self.shortcuts):
            x = F.relu(b(x) + s(x))
        return self.head(x.mean(-1))


def build_model(name, n_cls):
    if name == "eegnet":
        return EEGNet(1, WINDOW, n_cls, F1=32, D=2, F2=64), True   # [N,1,1,T]
    return HannunECGResNet(n_cls, n_blocks=N_BLOCKS), False        # [N,1,T]


def shape(is4d, x):
    return x[:, None, None, :] if is4d else x[:, None, :]


# ------------------------------------------------------------------ data
def load_cinc():
    """Return (signals [N, WINDOW] float32, labels [N] int64). Downloads
    ~200 MB from PhysioNet on first call."""
    if os.environ.get("MOCK", "0") == "1":
        rng = np.random.default_rng(0)
        X, y = [], []
        for lab in (0, 1):
            for _ in range(300):
                sig = rng.normal(0, .1, WINDOW); pos = 100
                while pos < WINDOW:
                    sig[pos:pos+8] += 3.0
                    pos += (220 + int(rng.integers(-8, 8))) if lab == 0 \
                           else int(rng.integers(140, 320))
                X.append(sig.astype("float32")); y.append(lab)
        return np.stack(X), np.array(y, dtype="int64")
    import scipy.io
    if not os.path.isdir("training2017"):
        os.system("wget -q -O training2017.zip "
                  "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip "
                  "|| wget -q -O training2017.zip "
                  "https://archive.physionet.org/challenge/2017/training2017.zip")
        os.system("unzip -qo training2017.zip")
    ref = pd.read_csv("training2017/REFERENCE.csv", header=None, names=["rec", "lab"])
    ref = ref[ref.lab.isin(["N", "A"])]
    n_af = int((ref.lab == "A").sum())
    ref = pd.concat([ref[ref.lab == "A"],
                     ref[ref.lab == "N"].sample(n=n_af, random_state=0)])
    X, y = [], []
    for rec, lab in zip(ref.rec, ref.lab):
        v = scipy.io.loadmat(f"training2017/{rec}.mat")["val"][0].astype("float32")
        v = (v - v.mean()) / (v.std() + 1e-8)          # per-recording standardization
        if len(v) < WINDOW:
            v = np.pad(v, (0, WINDOW - len(v)))
        c = (len(v) - WINDOW) // 2
        X.append(v[c:c+WINDOW]); y.append(1 if lab == "A" else 0)
    return np.stack(X), np.array(y, dtype="int64")


def train(model, X, y, epochs, seed, is4d):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            F.cross_entropy(model(shape(is4d, torch.tensor(X[idx])).to(DEV)),
                            torch.tensor(y[idx]).to(DEV)).backward()
            opt.step()
    return model.cpu().eval()


@torch.no_grad()
def accuracy(model, X, y, is4d):
    model.eval(); hits = 0
    for i in range(0, len(X), 64):
        hits += (model(shape(is4d, torch.tensor(X[i:i+64]))).argmax(1).numpy()
                 == y[i:i+64]).sum()
    return hits / len(y)


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
        done = {(r.model, r.split) for r in prev.itertuples()}
        print(f"resuming -- {len(done)} cells done")

    Xall, yall = load_cinc()
    print(f"CinC 2017: {len(Xall)} recordings, AF fraction {yall.mean():.2f}")

    for mname in MODELS:
        target = TARGETS[mname]
        for split in range(SPLITS):
            if (mname, split) in done:
                continue
            rng = np.random.default_rng(split)
            perm = rng.permutation(len(Xall))
            n_tr = int(0.65 * len(perm))
            n_sep = int(0.10 * len(perm))
            tr, sep, tg = perm[:n_tr], perm[n_tr:n_tr+n_sep], perm[n_tr+n_sep:]
            Xtr, ytr = Xall[tr], yall[tr]
            Xsep = Xall[sep]                       # held-out TRAINING distribution
            Xcal = Xall[tg[:N_CALIB]]              # unlabeled, new subjects
            Xev, yev = Xall[tg[N_CALIB:]], yall[tg[N_CALIB:]]

            model, is4d = build_model(mname, 2)
            dense = train(model, Xtr, ytr, EPOCHS, split, is4d)
            x4tr = shape(is4d, torch.tensor(Xtr[:256]))
            x4cal, x4sep = shape(is4d, torch.tensor(Xcal)), shape(is4d, torch.tensor(Xsep[:256]))
            reached = probe(dense, target, x4tr[:1]).reached
            acc_dense = accuracy(dense, Xev, yev, is4d)
            print(f"[{mname} split{split}] dense={acc_dense:.3f} "
                  f"repair={len(reached)} layers", flush=True)

            rows = []
            for ratio in RATIOS:
                if ratio == 0.0:
                    pruned, rec = copy.deepcopy(dense), list(reached)
                else:
                    pruned, rep = compress(dense, x4tr, torch.tensor(ytr[:256]),
                                           layer=target, ratio=ratio,
                                           criterion="discriminability")
                    rec = list(rep.recalibrate)
                d_tg, d_sep = delta(pruned, x4cal), delta(pruned, x4sep)
                tg_r, tg_u = split_delta(d_tg, rec)
                sp_r, sp_u = split_delta(d_sep, rec)
                m_p, m_f = copy.deepcopy(pruned), copy.deepcopy(pruned)
                if rec:
                    recalibrate(m_p, x4cal, layers=rec)
                recalibrate(m_f, x4cal)
                af_r, af_u = split_delta(delta(m_p, x4cal), rec)
                rows.append(dict(
                    domain="ecg", model=mname, split=split, layer=target, ratio=ratio,
                    n_bn=len(d_tg), n_recal=len(rec),
                    acc_dense=round(acc_dense, 4),
                    acc_none=round(accuracy(pruned, Xev, yev, is4d), 4),
                    acc_probe=round(accuracy(m_p, Xev, yev, is4d), 4),
                    acc_full=round(accuracy(m_f, Xev, yev, is4d), 4),
                    d_target_reached=round(tg_r, 4), d_target_untouched=round(tg_u, 4),
                    d_sep_reached=round(sp_r, 4), d_after_reached=round(af_r, 4),
                    delta_target=json.dumps({k: round(v, 4) for k, v in d_tg.items()})))
            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                      header=not os.path.exists(OUT))

    # ---------------------------------------------------------- summary
    df = pd.read_csv(OUT)
    from scipy.stats import wilcoxon
    for m, g in df.groupby("model"):
        print(f"\n=== ECG / {m}  (repair {g.n_recal.iloc[-1]}/{g.n_bn.iloc[-1]}, "
              f"dense {g.acc_dense.mean():.3f}) ===")
        print(f"{'ratio':>6} {'none':>7} {'probe':>7} {'full':>7} {'gain':>7} "
              f"{'Δrch':>7} {'Δunt':>7} {'Δsep':>7} {'Δaft':>7}")
        for r, gr in g.groupby("ratio"):
            print(f"{r:>6.3f} {gr.acc_none.mean():>7.3f} {gr.acc_probe.mean():>7.3f} "
                  f"{gr.acc_full.mean():>7.3f} {(gr.acc_probe-gr.acc_none).mean():>+7.3f} "
                  f"{gr.d_target_reached.mean():>7.3f} {gr.d_target_untouched.mean():>7.3f} "
                  f"{gr.d_sep_reached.mean():>7.3f} {gr.d_after_reached.mean():>7.3f}")
        lo, hi = g[g.ratio == 0.0], g[g.ratio == 0.875]
        gl = (lo.acc_probe-lo.acc_none).groupby(lo.split).mean()
        gh = (hi.acc_probe-hi.acc_none).groupby(hi.split).mean()
        d = gh - gl
        if d.std() > 0:
            print(f"  gain {gl.mean():+.3f} -> {gh.mean():+.3f} "
                  f"(increase {d.mean():+.3f}, {int((d>0).sum())}/{len(d)}, "
                  f"p={wilcoxon(d).pvalue:.3f})")
        sess = hi.d_target_reached.mean() - hi.d_sep_reached.mean()
        print(f"  separation @87.5%: pruning {hi.d_sep_reached.mean():.3f} | "
              f"subject-shift {sess:+.3f} ({sess/max(hi.d_target_reached.mean(),1e-9):.0%})")
        pf = (hi.acc_probe-hi.acc_full).groupby(hi.split).mean()
        print(f"  probe-set vs full AdaBN: {pf.mean():+.4f}" +
              (f" (p={wilcoxon(pf).pvalue:.3f})" if pf.std() > 0 else " (identical)"))


if __name__ == "__main__":
    main()
