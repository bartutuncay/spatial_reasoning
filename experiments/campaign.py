"""Collect worker results into a LEDGER (JSONL rewrite + Markdown table).

Idempotent: re-running ``collect`` fully rewrites the ledger from whatever
``result.json`` files exist. A spec with no result yet is PENDING. Each worker
writes ``results/<id>/result.json`` with at least::

    {"id": ..., "status": "ok"|"failed", "metrics": {...},
     "verdict": "EXISTS"|"WEAK"|"DEAD" (optional), "gpu_h": float (optional),
     "primary": <number or str> (optional), "notes": str (optional)}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# UNSCORED = the run completed but did not assert a control-based verdict, so it
# is NOT a green existence-proof. Only a worker that beats its matched control
# should emit EXISTS.
_RANK = {"DEAD": 0, "PENDING": 0, "UNSCORED": 0.5, "WEAK": 1, "FRAGILE": 1.5, "EXISTS": 2}


def _verdict(spec, res):
    if res is None:
        return "PENDING"
    if res.get("verdict") in _RANK:
        return res["verdict"]
    # a completed run without an explicit verdict is UNSCORED, never auto-EXISTS
    return "UNSCORED" if res.get("status") == "ok" else "DEAD"


def _load_result(results_dir: Path, sid: str):
    p = results_dir / sid / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):  # ValueError covers JSON + UnicodeDecodeError
        return {"status": "failed", "notes": "unreadable result.json"}


def collect(specs_dir, results_dir, out_dir=None):
    specs_dir = Path(specs_dir)
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for sp in sorted(specs_dir.glob("*.json")):
        spec = json.loads(sp.read_text())
        res = _load_result(results_dir, spec["id"]) or {}
        v = _verdict(spec, res or None)
        gpu_h = res.get("gpu_h")               # preserve a truthful 0.0
        if gpu_h is None:
            gpu_h = spec.get("est_gpu_h")
        rows.append({
            "id": spec["id"],
            "wave": spec.get("wave"),
            "channel": res.get("channel") or spec.get("channel"),
            "verdict": v,
            "primary": res.get("primary"),
            "gpu_h": gpu_h,
            "date": res.get("date") or today,  # worker's production date, preserved
            "hypothesis": spec.get("hypothesis", ""),
            "notes": res.get("notes", ""),
        })

    (out_dir / "ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (out_dir / "LEDGER.md").write_text(_render_md(rows))
    return rows


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    return "" if x is None else str(x)


def _render_md(rows):
    cols = ["id", "wave", "channel", "verdict", "primary", "gpu_h", "hypothesis"]
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |"
        for r in sorted(rows, key=lambda r: (r.get("wave", 0), r["id"]))
    ]
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f"# Spatial-JEPA campaign LEDGER\n\n_{len(rows)} specs: {summary}_\n\n" + "\n".join(
        [header, sep, *body]
    ) + "\n"


def main():
    import argparse

    from experiments.config import RESULTS_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="experiments/exp_jepa/specs")
    ap.add_argument("--results", default=RESULTS_DIR)
    args = ap.parse_args()
    rows = collect(args.specs, args.results)
    done = sum(1 for r in rows if r["verdict"] != "PENDING")
    print(f"ledger: {done}/{len(rows)} specs have results")


if __name__ == "__main__":
    main()
