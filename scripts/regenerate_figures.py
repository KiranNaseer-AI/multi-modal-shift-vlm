"""
Regenerate all 10 figures the paper references, with draft
annotations removed.

Writes to /content/figs (session storage, not Drive) and downloads
each one directly to your machine. Every figure is validated before
download -- any plot that came out empty is reported and NOT
offered, so a blank figure can never silently replace a good one.

Run as a single Colab cell after Drive is mounted.
"""

import json, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path("/content/drive/MyDrive/DAMF/logs")
OUT = Path("/content/figs")
OUT.mkdir(exist_ok=True)

STAGED = "Staged FT"
BALANCE = 1.3689
SEED_COLOR = {42: "tab:red", 0: "tab:blue", 123: "tab:green"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "legend.fontsize": 8, "figure.dpi": 150,
                     "savefig.bbox": "tight"})


def find_log(method, ds, seed):
    """Resolve the real filename rather than assuming a pattern."""
    for pat in (f"{method}_{ds}_seed{seed}.json",
                f"{method}_{ds}_seed_{seed}.json"):
        p = LOGS / pat
        if p.exists():
            return p
    hits = list(LOGS.glob(f"{method}_{ds}*{seed}.json"))
    return hits[0] if hits else None


def load(method, ds, seed):
    p = find_log(method, ds, seed)
    return json.load(open(p)) if p else None


def rt_series(method, ds, seed):
    d = load(method, ds, seed)
    if not d:
        return None, None
    rt = d.get("rt_log", [])
    if not rt:
        return None, None
    s = np.array([e[0] for e in rt], float)
    v = np.array([e[1] for e in rt], float)
    ok = np.isfinite(v) & (v > 0)
    return (s[ok], v[ok]) if ok.any() else (None, None)


print("Log availability check")
print("=" * 58)
NEEDED = ["naive_ft", "lowlr_ft", "damf", "joint_schedule_ft",
          "rms_balanced_ft", "isolated", "frozen_vis", "lora",
          "multimodal_lora"]
for ds in ("uicd", "rsicd", "rocov2"):
    have = [m for m in NEEDED if find_log(m, ds, 42)]
    miss = [m for m in NEEDED if m not in have]
    print(f"  {ds:7s} {len(have)}/9 present" +
          (f"   MISSING: {miss}" if miss else ""))
print()


def loss_vs_bleu(ds, fname, title_ds):
    panels = [("naive_ft", "Naive FT"), ("lowlr_ft", "Low-LR FT"),
              ("damf", STAGED)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    drew = 0
    for ax, (m, label) in zip(axes, panels):
        d = load(m, ds, 42)
        if d is None:
            ax.text(.5, .5, "no data", ha="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        loss, bleu = d["train_loss_per_epoch"], d["bleu4_per_epoch"]
        ep = range(1, len(loss) + 1)
        ax.plot(ep, loss, "o-", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        a2 = ax.twinx()
        a2.plot(ep, bleu, "s--", color="tab:green")
        a2.set_ylabel("BLEU-4", color="tab:green")
        a2.tick_params(axis="y", labelcolor="tab:green")
        if m == "damf":
            ax.axvline(2.5, color="navy", ls=":", lw=1)
            ax.text(2.55, ax.get_ylim()[1] * .9, "S1$\\to$S2",
                    fontsize=7, color="navy")
        ax.set_title(label)
        drew += 1
    fig.suptitle(f"{title_ds}, seed 42: training loss (red) vs "
                 f"BLEU-4 (green)", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    return drew


def rt_method_comparison(ds, fname, title_ds):
    spec = [("naive_ft", "Naive FT", "tab:red", "-"),
            ("lowlr_ft", "Low-LR FT", "tab:orange", "--"),
            ("damf", STAGED, "tab:green", "-")]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    drew = 0
    for m, label, c, ls in spec:
        grids, curves = [], []
        for s in (42, 0, 123):
            st, v = rt_series(m, ds, s)
            if st is None:
                continue
            grids.append(st)
            curves.append(v)
        if not curves:
            continue
        lo = max(g.min() for g in grids)
        hi = min(g.max() for g in grids)
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 250)
        stack = np.array([np.interp(grid, g, c_)
                          for g, c_ in zip(grids, curves)])
        mu, sd = stack.mean(0), stack.std(0)
        ax.plot(grid, mu, ls, color=c, lw=1.6,
                label=f"{label} (n={len(curves)})")
        ax.fill_between(grid, mu - sd, mu + sd, color=c, alpha=.15)
        drew += 1
    if drew:
        ax.axhline(BALANCE, color="gray", ls=":", lw=1)
        ax.text(ax.get_xlim()[1] * .55, BALANCE * 1.06,
                "per-parameter balance", fontsize=7, color="gray")
        ax.legend()
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("$r_t$  (seed-averaged, $\\pm$1 s.d.)")
    ax.set_title(f"{title_ds}: gradient-norm ratio during training")
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    return drew


def staged_rt_per_seed(ds, fname, title_ds):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    drew = 0
    for ax, s in zip(axes, (42, 0, 123)):
        st, v = rt_series("damf", ds, s)
        if st is None:
            ax.text(.5, .5, f"seed {s}: no $r_t$ data", ha="center",
                    transform=ax.transAxes)
            ax.set_title(f"Seed {s}")
            continue
        ax.plot(st, v, lw=.9, color=SEED_COLOR[s])
        ax.axhline(BALANCE, color="gray", ls="--", lw=.8)
        ax.set_xlabel("Optimiser step")
        ax.set_title(f"Seed {s}")
        drew += 1
    axes[0].set_ylabel("$r_t$")
    if drew:
        axes[0].text(.02, .94, "dashed: per-parameter balance",
                     transform=axes[0].transAxes, fontsize=6.5,
                     color="gray")
    fig.suptitle(f"{STAGED} on {title_ds}: $r_t$ during Stage 2", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    return drew


def stability(ds, fname, title_ds):
    spec = [("naive_ft", "Naive FT"), ("lowlr_ft", "Low-LR FT"),
            ("joint_schedule_ft", "Joint-Schedule FT"), ("damf", STAGED),
            ("rms_balanced_ft", "RMS-Balanced FT"),
            ("isolated", "Isolated Visual"),
            ("frozen_vis", "Frozen Vision"), ("lora", "Text-Only LoRA"),
            ("multimodal_lora", "Multimodal LoRA")]
    labels, best, final = [], [], []
    for m, lab in spec:
        b, f_ = [], []
        for s in (42, 0, 123):
            d = load(m, ds, s)
            if d is None:
                continue
            b.append(d["best_bleu4"])
            f_.append(d["bleu4_per_epoch"][-1])
        if not b:
            continue
        labels.append(lab)
        best.append(np.mean(b))
        final.append(np.mean(f_))
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    if labels:
        x = np.arange(len(labels))
        w = .38
        ax.bar(x - w / 2, best, w, label="Best epoch")
        ax.bar(x + w / 2, final, w, label="Final epoch",
               hatch="//", alpha=.75)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.legend()
    ax.set_ylabel("BLEU-4")
    ax.set_title(f"{title_ds}: best-epoch vs final-epoch BLEU-4")
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    return len(labels)


def staged_stages(ds, fname, title_ds):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    drew = 0
    for s in (42, 0, 123):
        d = load("damf", ds, s)
        if d is None:
            continue
        ep = range(1, len(d["bleu4_per_epoch"]) + 1)
        axes[0].plot(ep, d["bleu4_per_epoch"], "o-",
                     color=SEED_COLOR[s], label=f"Seed {s}")
        axes[1].plot(ep, d["val_loss_per_epoch"], "o-",
                     color=SEED_COLOR[s], label=f"Seed {s}")
        drew += 1
    for ax, ylab in zip(axes, ["BLEU-4", "Validation loss"]):
        ax.axvline(2.5, color="navy", ls=":", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        if drew:
            ax.legend(fontsize=7)
    fig.suptitle(f"{STAGED} on {title_ds}: two-stage training dynamics",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    return drew


JOBS = [
    ("fig01_uicd_loss_vs_bleu4.pdf",
     lambda f: loss_vs_bleu("uicd", f, "UICD")),
    ("fig05_rsicd_loss_vs_bleu4.pdf",
     lambda f: loss_vs_bleu("rsicd", f, "RSICD")),
    ("fig04_uicd_rt_method_comparison.pdf",
     lambda f: rt_method_comparison("uicd", f, "UICD")),
    ("fig09_rsicd_rt_method_comparison.pdf",
     lambda f: rt_method_comparison("rsicd", f, "RSICD")),
    ("fig14_rocov2_rt_method_comparison.pdf",
     lambda f: rt_method_comparison("rocov2", f, "ROCOv2")),
    ("fig03_uicd_rt_damf_per_seed.pdf",
     lambda f: staged_rt_per_seed("uicd", f, "UICD")),
    ("fig08_rsicd_rt_damf_per_seed.pdf",
     lambda f: staged_rt_per_seed("rsicd", f, "RSICD")),
    ("fig13_rocov2_rt_damf_per_seed.pdf",
     lambda f: staged_rt_per_seed("rocov2", f, "ROCOv2")),
    ("fig11_rocov2_stability.pdf",
     lambda f: stability("rocov2", f, "ROCOv2")),
    ("fig12_rocov2_damf_stages.pdf",
     lambda f: staged_stages("rocov2", f, "ROCOv2")),
]

print("Generating")
print("=" * 58)
good, bad = [], []
for fname, fn in JOBS:
    n = fn(fname)
    p = OUT / fname
    kb = p.stat().st_size / 1024 if p.exists() else 0
    if n == 0 or kb < 8:
        bad.append(fname)
        print(f"  EMPTY  {fname:42s} ({n} series, {kb:.0f} KB)")
    else:
        good.append(fname)
        print(f"  ok     {fname:42s} ({n} series, {kb:.0f} KB)")

subprocess.run(["apt-get", "install", "-qq", "poppler-utils"],
               capture_output=True)
print("\nDraft-annotation audit")
print("=" * 58)
dirty = []
for fname in good:
    txt = subprocess.run(["pdftotext", str(OUT / fname), "-"],
                         capture_output=True, text=True).stdout
    hits = [w for w in ("DAMF", "CONFIRMED", "Ours", "Anchors")
            if w.lower() in txt.lower()]
    if hits:
        dirty.append(fname)
        print(f"  DIRTY  {fname}  {hits}")
if not dirty:
    print("  all clean")

print("\nSummary")
print("=" * 58)
print(f"  ready to use : {len(good)}")
print(f"  empty        : {len(bad)}" + (f"  {bad}" if bad else ""))

if good:
    print(f"\nDownloading {len(good)} figures...")
    from google.colab import files
    for fname in good:
        files.download(str(OUT / fname))
    print("\nReplace these in your Overleaf figures/ folder.")
if bad:
    print(f"\nNOT downloaded (empty): {bad}")
    print("Tell me which and I will investigate before you replace anything.")
