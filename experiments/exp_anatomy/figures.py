"""Paper figures from the capability-anatomy result JSONs.

Fig1  capability matrix heatmap (rows x 5 columns, normalized floor->best).
Fig2  (a) dissociation scatter: discriminative composite vs predictive (rollout);
      (b) fusion frontier: placerec vs rollout, the contrastive<->jepa lambda-path.
Fig3  mechanism: I(z;appearance) vs discriminative composite (Spearman rho).

Colorblind-safe 2-pole scheme (contrastive=vermillion, jepa=blue), gray refs,
neutral others; direct labels everywhere (print PDF, no interactivity).
Usage: python -m experiments.exp_anatomy.figures <bulk_dir> <diag_dir> <out_dir>
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- palette (Okabe-Ito, CB-safe) ---
C_CONTRAST = "#D55E00"   # vermillion — discriminative pole
C_JEPA = "#0072B2"       # blue — predictive pole
C_FUSE = "#009E73"       # green — fusion path
C_REF = "#999999"        # gray — frozen foundation refs
C_OTHER = "#444444"      # dark ink — other trained rows
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

ROWS = ["scratch", "rgb_only", "recon", "symalign", "contrastive", "jepa",
        "fuse_cj_25", "fuse_cj_50", "fuse_cj_75",
        "ref:dinov2s", "ref:dinov2b", "ref:siglip", "ref:qwen2vl", "ref:vjepa2"]
DISP = {"scratch": "scratch", "rgb_only": "rgb-only", "recon": "recon",
        "symalign": "sym-align", "contrastive": "contrastive", "jepa": "jepa",
        "fuse_cj_25": "fuse.25", "fuse_cj_50": "fuse.50", "fuse_cj_75": "fuse.75",
        "ref:dinov2s": "DINOv2-S", "ref:dinov2b": "DINOv2-B",
        "ref:siglip": "SigLIP", "ref:qwen2vl": "Qwen2-VL", "ref:vjepa2": "V-JEPA2"}
# column, metric key, higher-is-better, floor key (None -> 0)
COLS = [("placerec", "top1_acc", True, "majority_floor"),
        ("depthprobe", "absrel", False, "floor_absrel"),
        ("rollout", "delta_r2", True, None),
        ("navdist", "r2", True, None),
        ("relpose", "dir_cos", True, "floor_dir_cos")]
COLLBL = ["place-rec", "depth", "rollout", "nav-dist", "rel-pose"]


def load(results_dir):
    agg = defaultdict(list)
    for f in glob.glob(f"{results_dir}/*/result.json"):
        if "pilot" in f or "/pil_" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        probe, _, row = d.get("channel", "").partition(":")
        agg[(probe, row)].append(d.get("metrics", {}))
    return agg


def cell(agg, probe, row, key):
    ms = [m[key] for m in agg.get((probe, row), []) if key in m and m[key] is not None]
    return float(np.mean(ms)) if ms else np.nan


def color_for(row):
    if row == "contrastive":
        return C_CONTRAST
    if row == "jepa":
        return C_JEPA
    if row.startswith("fuse"):
        return C_FUSE
    if row.startswith("ref:"):
        return C_REF
    return C_OTHER


def fig1_matrix(agg, out):
    rows = [r for r in ROWS if any((p, r) in agg for p, *_ in COLS)]
    M = np.full((len(rows), len(COLS)), np.nan)
    for j, (probe, key, hib, fk) in enumerate(COLS):
        vals, floors = {}, {}
        for r in rows:
            v = cell(agg, probe, r, key)
            fl = cell(agg, probe, r, fk) if fk else 0.0
            vals[r] = v; floors[r] = fl if fl == fl else 0.0
        fl = np.nanmean([floors[r] for r in rows])
        arr = np.array([vals[r] for r in rows])
        best = np.nanmax(arr) if hib else np.nanmin(arr)
        for i, r in enumerate(rows):
            v = vals[r]
            if v != v:
                continue
            M[i, j] = (v - fl) / (best - fl + 1e-9) if hib else (fl - v) / (fl - best + 1e-9)
    M = np.clip(M, 0, 1)
    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLLBL, rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([DISP[r] for r in rows])
    for i in range(len(rows)):
        for j in range(len(COLS)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=7)
    ax.axhline(8.5, color="w", lw=1.2)          # separate trained rows from refs
    ax.set_title("Capability (0=floor, 1=best-in-column)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    ax.grid(False)
    fig.savefig(out / "fig1_matrix.pdf"); plt.close(fig)


def fig2_dissociation(agg, out):
    rows = [r for r in ROWS if (("placerec", r) in agg and ("rollout", r) in agg)]
    pr = np.array([cell(agg, "placerec", r, "top1_acc") for r in rows])
    nd = np.array([cell(agg, "navdist", r, "r2") for r in rows])
    rp = np.array([cell(agg, "relpose", r, "dir_cos") for r in rows])
    ro = np.array([cell(agg, "rollout", r, "delta_r2") for r in rows])

    def z(a):
        return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
    disc = np.nanmean(np.vstack([z(pr), z(nd), z(rp)]), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    # (a) dissociation scatter
    for i, r in enumerate(rows):
        ax1.scatter(disc[i], ro[i], s=70 if r in ("contrastive", "jepa") else 40,
                    color=color_for(r), zorder=3,
                    edgecolor="k" if r in ("contrastive", "jepa") else "none", linewidth=0.7)
        if not r.startswith("ref:") or r in ("ref:dinov2b",):
            ax1.annotate(DISP[r], (disc[i], ro[i]), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
    ax1.set_xlabel("discriminative composite (z: place-rec, nav-dist, rel-pose)")
    ax1.set_ylabel("predictive  (rollout $\\Delta R^2$)")
    ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax1.set_title("(a) two capability axes")

    # (b) fusion frontier: placerec vs rollout, lambda-path
    path = ["jepa", "fuse_cj_25", "fuse_cj_50", "fuse_cj_75", "contrastive"]
    px = [cell(agg, "placerec", r, "top1_acc") for r in path]
    py = [cell(agg, "rollout", r, "delta_r2") for r in path]
    ax2.plot(px, py, "-", color=C_FUSE, lw=1.5, zorder=2)
    for r, x, y in zip(path, px, py):
        ax2.scatter(x, y, s=70, color=color_for(r), zorder=3, edgecolor="k", linewidth=0.6)
        ax2.annotate(DISP[r], (x, y), fontsize=7, xytext=(4, -2), textcoords="offset points")
    ax2.set_xlabel("place-rec accuracy $\\uparrow$")
    ax2.set_ylabel("rollout $\\Delta R^2$ $\\uparrow$")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax2.set_title("(b) fusion frontier ($\\lambda$: jepa$\\to$contrastive)")
    fig.tight_layout()
    fig.savefig(out / "fig2_dissociation.pdf"); plt.close(fig)


def fig3_mechanism(agg, diag, out):
    rows = [r for r in ROWS if (("placerec", r) in agg and ("diagnose", r) in diag)]
    iapp = np.array([cell(diag, "diagnose", r, "i_appear_nats") for r in rows])
    pr = np.array([cell(agg, "placerec", r, "top1_acc") for r in rows])
    ok = ~(np.isnan(iapp) | np.isnan(pr))
    x, y = iapp[ok], pr[ok]
    # Spearman rho
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for r, xi, yi in zip([r for r, o in zip(rows, ok) if o], x, y):
        ax.scatter(xi, yi, s=60 if r in ("contrastive", "jepa") else 40, color=color_for(r),
                   edgecolor="k" if r in ("contrastive", "jepa") else "none", linewidth=0.7, zorder=3)
        ax.annotate(DISP[r], (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("$I(z;\\,$appearance$)$  (nats)")
    ax.set_ylabel("place-rec accuracy")
    ax.set_title(f"mechanism: appearance info $\\to$ discrimination (Spearman $\\rho$={rho:.2f})",
                 fontsize=8.5)
    fig.savefig(out / "fig3_mechanism.pdf"); plt.close(fig)


def main():
    bulk_dir = sys.argv[1] if len(sys.argv) > 1 else "results/anatomy_bulk"
    diag_dir = sys.argv[2] if len(sys.argv) > 2 else "results/anatomy"
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "results/figures")
    out.mkdir(parents=True, exist_ok=True)
    agg = load(bulk_dir)
    diag = load(diag_dir)
    fig1_matrix(agg, out)
    fig2_dissociation(agg, out)
    fig3_mechanism(agg, diag, out)
    print(f"wrote fig1_matrix.pdf, fig2_dissociation.pdf, fig3_mechanism.pdf to {out}")


if __name__ == "__main__":
    main()
