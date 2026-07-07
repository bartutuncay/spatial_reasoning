import numpy as np
import torch

from experiments.exp_anatomy.common import (
    depth_grid, relative_action, seed_split, split_seeds, to_uint8, walk_sequences,
)


def test_seed_split_last_seed_eval():
    files = [f"x/rw_{s}_{i:03d}.pt" for s in (0, 1, 7) for i in range(3)]
    tr, ev = seed_split(files)
    assert all("rw_7_" in f for f in ev) and len(ev) == 3
    assert len(tr) == 6 and not any("rw_7_" in f for f in tr)


def test_depth_grid_masks_invalid():
    d = torch.zeros(64, 64)
    d[:32] = 2.0                                   # bottom half invalid (0)
    g, m = depth_grid(d, g=4)
    assert g.shape == (16,) and m.shape == (16,)
    assert m[:8].all() and not m[8:].any()
    assert torch.allclose(g[:8], torch.log(torch.tensor(2.0)).expand(8))


def test_depth_grid_nan_and_inf_are_invalid():
    d = torch.full((64, 64), float("nan"))
    d[:16] = 3.0
    d[16:32] = float("inf")
    g, m = depth_grid(d, g=4)
    assert not torch.isnan(g).any() and not torch.isinf(g).any()
    assert m[:4].all() and not m[4:].any()       # only the 3.0 rows valid
    assert torch.allclose(g[:4], torch.log(torch.tensor(3.0)).expand(4))


def test_to_uint8_handles_both_ranges():
    a = to_uint8(np.ones((4, 4, 3), dtype=np.float32) * 0.5)     # 0-1 range
    b = to_uint8(np.ones((4, 4, 3), dtype=np.float32) * 200.0)   # 0-255 range
    assert a.dtype == np.uint8 and 126 <= a[0, 0, 0] <= 129
    assert b.dtype == np.uint8 and b[0, 0, 0] == 200


def test_walk_sequences_numeric_step_order(tmp_path):
    d = tmp_path / "office" / "random_walks"
    d.mkdir(parents=True)
    for step in (0, 2, 10):                            # lexicographic puts 10 < 2
        torch.save({"loc": [float(step), 0.0, 0.0]}, d / f"rw_0_{step}.pt")
    seqs = walk_sequences(str(tmp_path), "office")
    assert list(seqs.keys()) == [0]
    steps = [int(f.split("_")[-1].split(".")[0]) for f in seqs[0]]
    assert steps == [0, 2, 10]


def test_split_seeds_holds_out_top():
    seqs = {s: [f"rw_{s}_{i}.pt" for i in range(3)] for s in range(10)}
    tr, ev = split_seeds(seqs, n_eval=3)
    assert sorted(ev) == [7, 8, 9] and len(tr) == 7


def test_relative_action_world_delta():
    si = {"loc": [0.0, 0.0, 0.0], "view_dir": [1.0, 0.0, 0.0]}
    sj = {"loc": [1.0, 2.0, 0.0], "view_dir": [0.0, 1.0, 0.0]}
    a = relative_action(si, sj)
    assert a.shape == (6,)
    assert np.allclose(a[:3], [1.0, 2.0, 0.0])
    assert np.allclose(a[3:], [-1.0, 1.0, 0.0])
