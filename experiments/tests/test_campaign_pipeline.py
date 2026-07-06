"""End-to-end test of the campaign machinery with a dummy worker.

Exercises make_specs -> gen_sbatch -> (fake results) -> campaign.collect
without torch or a cluster, proving the pipeline before it drives real jobs.
"""
import json

from experiments import campaign, gen_sbatch, make_specs


def test_specs_are_well_formed_and_unique():
    specs = make_specs.build_specs()
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids))
    required = {"id", "wave", "hypothesis", "channel", "module", "args", "seed", "slurm"}
    for s in specs:
        assert required <= set(s)
    # the four-arm killer ablation must be present as an axis sweep
    objectives = {
        s["args"].get("--objective") for s in specs if s["channel"] == "objective"
    }
    objectives.add(make_specs.DEFAULTS["objective"])  # anchor carries the default
    assert {"jepa", "symalign", "recon", "contrastive"} <= objectives
    # all five day-1 probes present
    assert {f"W0_P{i}" for i in range(1, 6)} <= set(ids)


def test_sbatch_renders_expected_header_and_command():
    anchor = next(s for s in make_specs.build_specs() if s["id"] == "W1_anchor")
    txt = gen_sbatch.render(anchor)
    assert "#SBATCH --job-name=sj_W1_anchor" in txt
    assert "#SBATCH --gpus=nvidia_geforce_rtx_4090:1" in txt
    assert "#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa" in txt
    assert "python -m experiments.exp_jepa.train" in txt
    assert "--out results/jepa_campaign/W1_anchor" in txt   # write path == collector read path
    assert "--scene office" in txt                          # explicit scene, no silent default
    assert "--seed 0" in txt
    assert "source experiments/slurm/common_setup.sh" in txt


def test_collect_builds_ledger_with_verdicts(tmp_path):
    specs = make_specs.build_specs()
    specs_dir = tmp_path / "specs"
    make_specs.write_specs(specs, specs_dir)
    results_dir = tmp_path / "results"

    # simulate three finished workers: ok-without-verdict -> UNSCORED (never
    # auto-EXISTS); explicit EXISTS honored; failed -> DEAD.
    unscored_id, exists_id, fail_id = "W0_P3", "W0_P1", "W1_anchor"
    (results_dir / unscored_id).mkdir(parents=True)
    (results_dir / unscored_id / "result.json").write_text(json.dumps(
        {"id": unscored_id, "status": "ok", "primary": 0.0, "gpu_h": 0.0,
         "date": "2026-07-06", "metrics": {"centroid_ate": 0.9}}
    ))
    (results_dir / exists_id).mkdir(parents=True)
    (results_dir / exists_id / "result.json").write_text(json.dumps(
        {"id": exists_id, "status": "ok", "verdict": "EXISTS", "primary": 0.05}
    ))
    (results_dir / fail_id).mkdir(parents=True)
    (results_dir / fail_id / "result.json").write_text(json.dumps(
        {"id": fail_id, "status": "failed", "notes": "cuda oom"}
    ))

    rows = campaign.collect(specs_dir, results_dir, out_dir=tmp_path / "out")

    assert len(rows) == len(specs)
    by_id = {r["id"]: r for r in rows}
    assert by_id[unscored_id]["verdict"] == "UNSCORED"      # no auto-EXISTS
    assert by_id[unscored_id]["primary"] == 0.0
    assert by_id[unscored_id]["gpu_h"] == 0.0               # truthful 0.0 preserved, not estimate
    assert by_id[unscored_id]["date"] == "2026-07-06"       # worker date preserved
    assert by_id[exists_id]["verdict"] == "EXISTS"          # explicit verdict honored
    assert by_id[fail_id]["verdict"] == "DEAD"
    assert by_id["W0_P2"]["verdict"] == "PENDING"           # no result yet

    # ledger files written and idempotent (row count == spec count)
    ledger = (tmp_path / "out" / "ledger.jsonl").read_text().strip().splitlines()
    assert len(ledger) == len(specs)
    md = (tmp_path / "out" / "LEDGER.md").read_text()
    assert "# Spatial-JEPA campaign LEDGER" in md
    assert "UNSCORED=1" in md and "EXISTS=1" in md and "DEAD=1" in md
