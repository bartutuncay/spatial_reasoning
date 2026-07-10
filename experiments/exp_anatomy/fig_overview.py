"""Teaser / graphical abstract for the capability-anatomy paper.

A wide banner (spans both CVPR columns): rendered RGB inputs + point-cloud graph
-> shared latent -> conditioning sweep (the rows) -> frozen probe battery (the
columns) -> the double-dissociation result. Real renders are read from an assets
dir; result numbers are the ETH3D bulk matrix (hardcoded, traceable to the tally).

Usage: python -m experiments.exp_anatomy.fig_overview [assets_dir] [out.pdf]
  Defaults: assets_dir = <here>/assets (committed rendered frames),
            out.pdf    = spatial_reasoning/paper/figures/fig0_overview.pdf.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Okabe-Ito palette (matches the other paper figures)
C_CONTRAST = "#D55E00"
C_JEPA = "#0072B2"
C_FUSE = "#009E73"
C_GREY = "#999999"
C_BOX = "#2b2b2b"
C_LAT = "#444444"

plt.rcParams.update({
    "font.size": 7.2, "font.family": "sans-serif",
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def rbox(ax, x, y, w, h, fc="white", ec=C_BOX, lw=0.9, r=0.02, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=0.5))


def arrow(ax, x0, y0, x1, y1, color=C_LAT, lw=1.3, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, color=color, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


def graph_glyph(ax, cx, cy, s=1.0):
    """Small stylized point-cloud graph: nodes + edges."""
    rng = np.random.RandomState(3)
    pts = rng.rand(15, 2) - 0.5
    pts[:, 0] *= 7 * s
    pts[:, 1] *= 7 * s
    pts += [cx, cy]
    # a few edges (nearest-ish)
    for i in range(len(pts)):
        d = np.hypot(*(pts - pts[i]).T)
        for j in np.argsort(d)[1:3]:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=C_JEPA, lw=0.5, alpha=0.55, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=11, c=C_JEPA, zorder=3,
               edgecolors="white", linewidths=0.4)


def rgb_inset(fig, ax, img_path, x, y, w, h, label):
    """Place a rendered frame at data-coords (x,y,w,h) on the 0..100 axes."""
    tx = ax.transData
    # convert data rect to figure fraction
    (fx0, fy0) = fig.transFigure.inverted().transform(tx.transform((x, y)))
    (fx1, fy1) = fig.transFigure.inverted().transform(tx.transform((x + w, y + h)))
    iax = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0], zorder=4)
    iax.imshow(mpimg.imread(img_path))
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(C_BOX); s.set_linewidth(0.8)
    iax.set_xlabel(label, fontsize=6.3, labelpad=1.5)


def main(assets, out):
    fig = plt.figure(figsize=(7.0, 2.35))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- section captions ----
    for xc, t in [(11, "rendered inputs"), (31.5, "one embedding space"),
                  (52, "vary the objective"), (72, "frozen probes"),
                  (89.5, "capability")]:
        ax.text(xc, 97, t, ha="center", va="top", fontsize=7.4, weight="bold", color=C_BOX)

    # ---- (1) rendered inputs: two real RGB frames + a point-cloud graph glyph ----
    rgb_inset(fig, ax, f"{assets}/eth_rgb.png", 2.5, 62, 8.2, 21, "ETH3D")
    rgb_inset(fig, ax, f"{assets}/scn_rgb.png", 11.3, 62, 8.2, 21, "ScanNet")
    rbox(ax, 2.5, 22, 17, 27, fc="#f5f9fd", ec="#cfe0ef", lw=0.8, r=0.03, z=1)
    graph_glyph(ax, 11, 36.5, s=2.1)
    ax.text(11, 18.5, "point-cloud graph", ha="center", fontsize=6.2,
            color=C_JEPA, weight="bold")

    # ---- (2) shared latent ----
    arrow(ax, 20, 52, 27, 52)
    rbox(ax, 27, 40, 12, 26, fc="#f3f3f3")
    ax.text(33, 58, "shared", ha="center", fontsize=7.0, weight="bold")
    ax.text(33, 53, "128-d", ha="center", fontsize=7.0, weight="bold")
    ax.text(33, 47.5, "latent", ha="center", fontsize=7.0, weight="bold")
    ax.text(33, 43, "RGBenc+GNN", ha="center", fontsize=5.6, color=C_LAT)

    # ---- (3) conditioning sweep (rows) ----
    arrow(ax, 39.5, 52, 44, 52)
    chips = [("scratch", C_GREY), ("recon", C_GREY), ("sym-align", C_GREY),
             ("rgb-only", C_GREY), ("contrastive", C_CONTRAST),
             ("jepa", C_JEPA), ("fuse(λ)", C_FUSE)]
    y0 = 82
    for i, (name, col) in enumerate(chips):
        yy = y0 - i * 9.6
        rbox(ax, 44.5, yy - 3.4, 15, 6.6, fc=col if col != C_GREY else "#eeeeee",
             ec=col, lw=1.0, r=0.03)
        ax.text(52, yy, name, ha="center", va="center", fontsize=6.3,
                color="white" if col != C_GREY else C_BOX,
                weight="bold" if col != C_GREY else "normal")

    # ---- (4) probe battery (columns) ----
    for i in range(7):
        yy = y0 - i * 9.6
        arrow(ax, 59.8, yy, 64.5, 50, color="#cccccc", lw=0.5, z=1)
    probes = ["place-rec", "depth", "rollout", "nav-dist", "rel-pose"]
    py0 = 74
    for i, p in enumerate(probes):
        yy = py0 - i * 11
        rbox(ax, 65, yy - 3.8, 15.5, 7.4, fc="white", ec=C_BOX, lw=0.8, r=0.03)
        ax.text(72.7, yy, p, ha="center", va="center", fontsize=6.6)

    # ---- (5) result: double dissociation ----
    arrow(ax, 81, 50, 85, 50)
    # mini scatter inside a framed region
    rx, ry, rw, rh = 84.5, 18, 14.5, 64
    rbox(ax, rx, ry, rw, rh, fc="white", ec=C_BOX, lw=0.8, r=0.02)
    # data: (placerec, rollout) ETH3D bulk
    pts = {"contrastive": (0.852, -0.093, C_CONTRAST), "jepa": (0.522, 0.355, C_JEPA),
           "fuse": (0.792, 0.275, C_FUSE), "recon": (0.748, 0.142, C_GREY),
           "scratch": (0.217, 0.196, C_GREY)}
    xs = [v[0] for v in pts.values()]; ys = [v[1] for v in pts.values()]
    def X(v): return rx + 2.4 + (v - 0.15) / (0.95 - 0.15) * (rw - 4.4)
    def Y(v): return ry + 6 + (v + 0.15) / (0.40 + 0.15) * (rh - 12)
    for name, (px, py, col) in pts.items():
        big = name in ("contrastive", "jepa")
        ax.scatter([X(px)], [Y(py)], s=42 if big else 16, c=col, zorder=6,
                   edgecolors="white", linewidths=0.5)
    ax.annotate("", xy=(X(0.522), Y(0.355)), xytext=(X(0.852), Y(-0.093)),
                arrowprops=dict(arrowstyle="<->", color=C_BOX, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"), zorder=5)
    ax.text(X(0.80), Y(-0.09) - 3.2, "contrastive", ha="center", fontsize=5.8,
            color=C_CONTRAST, weight="bold")
    ax.text(X(0.50), Y(0.36) + 3.0, "jepa", ha="center", fontsize=5.8,
            color=C_JEPA, weight="bold")
    ax.text(rx + rw / 2, ry + rh - 3.2, "predictive ↑", ha="center",
            fontsize=5.6, color=C_LAT)
    ax.text(rx + rw / 2, ry + 2.6, "discriminative →", ha="center",
            fontsize=5.6, color=C_LAT)

    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    assets = sys.argv[1] if len(sys.argv) > 1 else str(_here / "assets")
    out = sys.argv[2] if len(sys.argv) > 2 else str(
        _here.parents[2] / "paper" / "figures" / "fig0_overview.pdf")
    main(assets, out)
