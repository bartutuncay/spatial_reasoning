"""Tally wave-1 anatomy probe results into a capability matrix (stdout + JSONL)."""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROWS_ORDER = ["scratch", "recon", "symalign", "contrastive", "jepa",
              "ref:dinov2s", "ref:dinov2b", "ref:siglip", "ref:qwen2vl"]


def load():
    out = []
    for f in sorted(glob.glob("results/anatomy/an_*/result.json")):
        d = json.load(open(f))
        out.append(d)
    return out


def main():
    rows = load()
    print(f"# Wave-1 anatomy tally: {len(rows)} result files\n")

    dead = [d for d in rows if d.get("status") != "ok"]
    if dead:
        print(f"## Non-ok jobs ({len(dead)})")
        for d in dead:
            print(f"  {d.get('id')}: {d.get('verdict')} — {d.get('notes','')[:120]}")
        print()

    # group by (probe channel prefix, row) -> average over seeds
    agg = defaultdict(list)  # (probe, row) -> list of metrics dicts
    for d in rows:
        if d.get("status") != "ok":
            continue
        ch = d.get("channel", "")           # e.g. "placerec:scratch" or "placerec:ref:dinov2s"
        probe, _, row = ch.partition(":")
        agg[(probe, row)].append(d)

    for probe in ("placerec", "depthprobe"):
        keys = [k for k in agg if k[0] == probe]
        if not keys:
            continue
        print(f"## {probe}")
        # header per probe
        if probe == "placerec":
            print(f"  {'row':14s} | {'acc':>7s} | {'floor':>7s} | {'NN@1':>7s} | n_seeds")
        else:
            print(f"  {'row':14s} | {'AbsRel':>7s} | {'floor':>7s} | {'d1.25':>7s} | n_seeds")
        ordered = sorted(keys, key=lambda k: ROWS_ORDER.index(k[1]) if k[1] in ROWS_ORDER else 99)
        for (_, row) in ordered:
            ds = agg[(probe, row)]
            ms = [d["metrics"] for d in ds]
            if probe == "placerec":
                acc = sum(m["top1_acc"] for m in ms) / len(ms)
                fl = sum(m["majority_floor"] for m in ms) / len(ms)
                nn = sum(m["nn_recall1"] for m in ms) / len(ms)
                print(f"  {row:14s} | {acc:7.3f} | {fl:7.3f} | {nn:7.3f} | {len(ds)}")
            else:
                ar = sum(m["absrel"] for m in ms) / len(ms)
                fl = sum(m["floor_absrel"] for m in ms) / len(ms)
                dd = sum(m["delta125"] for m in ms) / len(ms)
                print(f"  {row:14s} | {ar:7.3f} | {fl:7.3f} | {dd:7.3f} | {len(ds)}")
        print()

    # dump machine-readable JSONL
    Path("results/anatomy").mkdir(parents=True, exist_ok=True)
    with open("results/anatomy/wave1_tally.jsonl", "w") as fh:
        for d in rows:
            fh.write(json.dumps({k: d.get(k) for k in
                     ("id", "channel", "status", "verdict", "primary", "metrics")}) + "\n")
    print("wrote results/anatomy/wave1_tally.jsonl")


if __name__ == "__main__":
    main()
