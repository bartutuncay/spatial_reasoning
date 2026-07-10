"""Fig. 4 — dissecting the two axes (from the committed tally JSONLs).

(a) rollout delta-R2 vs horizon k: jepa is FLAT (slow features), everything else
    rises with k; rgb-only (rank-collapsed) rises steepest.
(b) training-length curves, rollout: contrastive converges to jepa by 15k steps.
(c) training-length curves, rel-pose: the relational gap WIDENS with training.

Usage: python -m experiments.exp_anatomy.fig_dissection [tally_dir] [out.pdf]
Defaults: tally_dir = <here>/tallies, out = paper/figures/fig4_dissection.pdf.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"contrastive": "#D55E00", "jepa": "#0072B2", "rgb_only": "#CC79A7",
     "scratch": "#999999", "fuse_cj_50": "#009E73", "ref:dinov2b": "#666666"}
LBL = {"contrastive": "contrastive", "jepa": "jepa", "rgb_only": "rgb-only",
       "scratch": "scratch", "fuse_cj_50": "fuse.50", "ref:dinov2b": "DINOv2-B"}
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "pdf.fonttype": 42})


def load(p):
    acc = defaultdict(list)
    for line in open(p):
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("primary") is not None:
            acc[r["channel"]].append(float(r["primary"]))
    return {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) for k, v in acc.items()}


def main(tdir, out):
    tdir = Path(tdir)
    bulk = load(tdir / "anatomy_bulk.jsonl")
    hk = {1: load(tdir / "anatomy_horizon_k1.jsonl"),
          4: load(tdir / "anatomy_horizon_k4.jsonl"),
          8: load(tdir / "anatomy_horizon_k8.jsonl")}
    c5 = load(tdir / "anatomy_curves_st5000.jsonl")
    c15 = load(tdir / "anatomy_curves_st15000.jsonl")

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))

    # (a) horizon sweep
    ks = [1, 2, 4, 8]
    for row in ["jepa", "contrastive", "rgb_only", "scratch", "ref:dinov2b"]:
        ch = f"rollout:{row}"
        ys, es = zip(*[(hk[k][ch] if k != 2 else bulk[ch]) for k in ks])
        ax[0].errorbar(ks, [y for y in ys], yerr=[e for e in es], color=C[row],
                       marker="o", ms=3, lw=1.4, capsize=2, label=LBL[row])
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks)
    ax[0].set_xlabel("rollout horizon $k$ (steps)")
    ax[0].set_ylabel(r"rollout $\Delta R^2$")
    ax[0].set_title("(a) jepa is flat in horizon", fontsize=8)
    ax[0].legend(fontsize=6, frameon=False, loc="upper left", ncol=2)

    # (b,c) training curves
    steps = [1500, 5000, 15000]
    for j, (probe, ylab, title) in enumerate(
            [("rollout", r"rollout $\Delta R^2$", "(b) rollout converges"),
             ("relpose", "rel-pose dir-cos", "(c) relational gap widens")], start=1):
        for row in ["contrastive", "jepa"]:
            ch = f"{probe}:{row}"
            ys, es = zip(*[bulk[ch], c5[ch], c15[ch]])
            ax[j].errorbar(steps, ys, yerr=es, color=C[row], marker="o", ms=3,
                           lw=1.4, capsize=2, label=LBL[row])
        ax[j].set_xscale("log"); ax[j].set_xticks(steps)
        ax[j].set_xticklabels(["1.5k", "5k", "15k"])
        ax[j].set_xlabel("conditioning steps")
        ax[j].set_ylabel(ylab)
        ax[j].set_title(title, fontsize=8)
        ax[j].legend(fontsize=6.5, frameon=False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    tdir = sys.argv[1] if len(sys.argv) > 1 else here / "tallies"
    out = sys.argv[2] if len(sys.argv) > 2 else str(
        here.parents[2] / "paper" / "figures" / "fig4_dissection.pdf")
    main(tdir, out)
