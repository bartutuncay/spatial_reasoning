"""Self-contained sanity checks for the JEPA core (no pytest needed).

Run in the env:  python -m experiments.exp_jepa.smoke_jepa
Exits 0 on success, non-zero (with a message) on the first failed assertion.
"""
import torch

from experiments.exp_jepa.jepa import (
    Predictor,
    clone_as_target,
    effective_rank,
    ema_update,
    info_nce,
    jepa_objective,
    vicreg_terms,
)


def _close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def check_predictor_shape():
    p = Predictor(dim=16, depth=2)
    x = torch.randn(8, 16)
    assert p(x).shape == (8, 16)


def check_ema_moves_target():
    online = torch.nn.Linear(4, 4)
    target = clone_as_target(online)
    before = next(target.parameters()).clone()
    with torch.no_grad():
        for prm in online.parameters():
            prm.add_(1.0)  # perturb online
    ema_update(target, online, momentum=0.9)
    after = next(target.parameters())
    assert not torch.allclose(before, after), "EMA target did not move toward online"
    # target params must never require grad
    assert all(not prm.requires_grad for prm in target.parameters())


def check_vicreg_detects_collapse():
    collapsed = torch.zeros(32, 8) + torch.randn(1, 8)   # all rows identical
    spread = torch.randn(32, 8)
    v_col, _ = vicreg_terms(collapsed)
    v_spr, _ = vicreg_terms(spread)
    assert v_col > v_spr, "variance term should punish collapse more"


def check_info_nce_rewards_alignment():
    z = torch.randn(16, 8)
    aligned = info_nce(z, z.clone())
    scrambled = info_nce(z, z[torch.randperm(16)])
    assert aligned < scrambled, "InfoNCE should be lower for aligned pairs"


def check_effective_rank():
    rank1 = torch.randn(64, 1) @ torch.randn(1, 10)   # rank-1 batch
    full = torch.randn(64, 10)
    r1, rf = effective_rank(rank1), effective_rank(full)
    assert r1 < 2.0, f"rank-1 batch effective rank too high: {r1}"
    assert rf > r1, "full-rank batch should have higher effective rank"


def check_all_objectives_finite():
    z_ctx = torch.randn(16, 32, requires_grad=True)
    z_tgt = torch.randn(16, 32)
    pred = Predictor(32)
    for obj in ("jepa", "symalign", "contrastive", "recon"):
        loss, terms = jepa_objective(z_ctx, z_tgt, predictor=pred, objective=obj)
        assert torch.isfinite(loss), f"{obj} loss not finite"
        assert "rank_ctx" in terms
        if obj != "recon":
            loss.backward(retain_graph=True)  # gradient flows to context
    # jepa without a predictor must error
    try:
        jepa_objective(z_ctx, z_tgt, predictor=None, objective="jepa")
        raise AssertionError("jepa without predictor should assert")
    except AssertionError as e:
        if "predictor" not in str(e):
            raise


def main():
    checks = [
        check_predictor_shape,
        check_ema_moves_target,
        check_vicreg_detects_collapse,
        check_info_nce_rewards_alignment,
        check_effective_rank,
        check_all_objectives_finite,
    ]
    for c in checks:
        c()
        print(f"  ok: {c.__name__}")
    print("JEPA CORE SMOKE: ALL PASSED")


if __name__ == "__main__":
    main()
