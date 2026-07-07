import numpy as np
import torch

from experiments.exp_anatomy.common import depth_grid, seed_split, to_uint8


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


def test_to_uint8_handles_both_ranges():
    a = to_uint8(np.ones((4, 4, 3), dtype=np.float32) * 0.5)     # 0-1 range
    b = to_uint8(np.ones((4, 4, 3), dtype=np.float32) * 200.0)   # 0-255 range
    assert a.dtype == np.uint8 and 126 <= a[0, 0, 0] <= 129
    assert b.dtype == np.uint8 and b[0, 0, 0] == 200
