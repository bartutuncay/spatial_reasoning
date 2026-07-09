"""Tally ALL anatomy probe results (Wave 1 + 2) into the capability matrix.

Prints one block per capability column with rows averaged over seeds, and writes
results/anatomy/matrix_tally.jsonl. Column verdict conventions:
  placerec   : linear-probe acc (floor 0.20)          higher better
  depthprobe : AbsRel (train-mean floor)              lower  better
  rollout    : delta-R^2 (copy-last floor = 0)        higher better
  navdist    : R^2 (mean-distance floor = 0)          higher better
  relpose    : direction cosine (mean-dir floor)      higher better
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROWS_ORDER = ["scratch", "rgb_only", "recon", "symalign", "contrastive", "jepa",
              "ref:dinov2s", "ref:dinov2b", "ref:siglip", "ref:qwen2vl"]

# probe -> (metric key, floor key or None, secondary key or None)
COLS = {
    "placerec":   ("top1_acc", "majority_floor", "nn_recall1"),
    "depthprobe": ("absrel", "floor_absrel", "delta125"),
    "rollout":    ("delta_r2", "copylast_floor_r2", "action_gap"),
    "rollout_act": ("delta_r2", "copylast_floor_r2", "action_gap"),  # branching walks
    "navdist":    ("r2", None, "mae_m"),
    "relpose":    ("dir_cos", "floor_dir_cos", "heading_mae"),
}


def load(results_dir="results/anatomy"):
    out = []
    for f in sorted(glob.glob(f"{results_dir}/*/result.json")):
        if "pilot" in f or "/pil_" in f:   # skip pilot runs (contaminate aggregation)
            continue
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def rank(row):
    return ROWS_ORDER.index(row) if row in ROWS_ORDER else 99


def main():
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/anatomy"
    rows = load(results_dir)
    print(f"# Anatomy capability matrix ({results_dir}): {len(rows)} result files\n")

    dead = [d for d in rows if d.get("status") != "ok"]
    if dead:
        print(f"## Non-ok jobs ({len(dead)})")
        for d in dead:
            print(f"  {d.get('id')}: {d.get('verdict')} — {d.get('notes','')[:110]}")
        print()

    agg = defaultdict(list)  # (probe, row) -> [metrics dict]
    for d in rows:
        if d.get("status") != "ok":
            continue
        probe, _, row = d.get("channel", "").partition(":")
        agg[(probe, row)].append(d.get("metrics", {}))

    for probe, (mk, fk, sk) in COLS.items():
        keys = sorted([k for k in agg if k[0] == probe], key=lambda k: rank(k[1]))
        if not keys:
            continue
        print(f"## {probe}  (metric={mk}, floor={fk}, extra={sk})")
        print(f"  {'row':14s} | {mk:>10s} | {'floor':>8s} | {sk or '':>10s} | seeds")
        for (_, row) in keys:
            ms = agg[(probe, row)]

            def avg(key):
                vals = [m[key] for m in ms if key in m and m[key] is not None]
                return sum(vals) / len(vals) if vals else float("nan")

            mval = avg(mk)
            fval = avg(fk) if fk else float("nan")
            sval = avg(sk) if sk else float("nan")
            print(f"  {row:14s} | {mval:10.3f} | {fval:8.3f} | {sval:10.3f} | {len(ms)}")
        print()

    # A3 mechanism diagnostics (not a capability column; explains the axes)
    dkeys = sorted([k for k in agg if k[0] == "diagnose"], key=lambda k: rank(k[1]))
    if dkeys:
        cols = ["i_pose_nats", "i_appear_nats", "uniformity", "alignment",
                "eff_rank", "smoothness"]
        hdr = ["i_pose", "i_appr", "unif", "align", "rank", "smooth"]
        print("## diagnose  (mechanism; higher I = more decodable)")
        print("  " + f"{'row':14s} | " + " | ".join(f"{h:>7s}" for h in hdr))
        for (_, row) in dkeys:
            ms = agg[("diagnose", row)]

            def davg(key):
                vals = [m[key] for m in ms if key in m and m[key] is not None
                        and m[key] == m[key]]
                return sum(vals) / len(vals) if vals else float("nan")

            cells = " | ".join(f"{davg(c):7.3f}" for c in cols)
            print(f"  {row:14s} | {cells}")
        print()

    # Write the JSONL INTO the results-dir it read, so datasets don't clobber
    # each other (results/anatomy/, results/anatomy_bulk/, results/anatomy_scannet/).
    out_jsonl = f"{results_dir.rstrip('/')}/matrix_tally.jsonl"
    with open(out_jsonl, "w") as fh:
        for d in rows:
            fh.write(json.dumps({k: d.get(k) for k in
                     ("id", "channel", "status", "verdict", "primary", "metrics")}) + "\n")
    print(f"wrote {out_jsonl}")


if __name__ == "__main__":
    main()
