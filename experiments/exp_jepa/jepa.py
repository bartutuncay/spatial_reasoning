"""The JEPA objective core, decoupled from the base encoders.

Given a context latent ``z_ctx`` (from the RGB image) and a target latent
``z_tgt`` (from the point-cloud graph, produced by an EMA/stop-grad target
encoder), ``--objective`` selects how they relate during training. This is the
four-arm killer ablation, driven from one place:

  jepa        : predictor(z_ctx) -> stopgrad(z_tgt), + VICReg anti-collapse
  symalign    : MSE(z_ctx, z_tgt), NO predictor / NO EMA  (the anti-reskin control)
  contrastive : InfoNCE(z_ctx, z_tgt)
  recon       : returns 0 here; the base pixel/depth decoders carry it in the trainer

The definitional JEPA pieces (asymmetric predictor + EMA target + VICReg) must be
shown to BEAT ``symalign`` (which is the model's current loss with decoders
removed) or the label is naming, not method.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """Asymmetric predictor on the context side (a JEPA definitional piece)."""

    def __init__(self, dim: int, hidden: int | None = None, depth: int = 2):
        super().__init__()
        hidden = hidden or dim * 2
        layers, d = [], dim
        # depth = number of Linear layers; depth==1 is a genuinely shallow
        # (single linear) predictor, so the shallow-vs-deep ablation is real.
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def clone_as_target(module: nn.Module) -> nn.Module:
    """A stop-grad copy of ``module`` to serve as the EMA target encoder."""
    tgt = copy.deepcopy(module)
    for p in tgt.parameters():
        p.requires_grad_(False)
    tgt.eval()
    return tgt


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """target <- momentum*target + (1-momentum)*online (params); copy buffers."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4, std_target: float = 1.0):
    """VICReg variance + covariance anti-collapse regularizers for a batch (N,D)."""
    z = z - z.mean(dim=0, keepdim=True)
    # biased variance (divide by N, not N-1) so batch size 1 gives 0, not NaN
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(std_target - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss, cov_loss


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1):
    """Symmetric InfoNCE with in-batch negatives (paired rows are positives)."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def effective_rank(z: torch.Tensor) -> float:
    """Effective rank (exp of spectral entropy) — a collapse diagnostic.

    ~1 means the batch collapsed to one direction; ~D means full spread.
    """
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z.float())
    p = s / (s.sum() + 1e-12)
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(entropy))


def jepa_objective(z_ctx, z_tgt, predictor=None, objective="jepa",
                   vicreg_w=(1.0, 0.04), temperature=0.1):
    """Return (loss, terms) for the chosen objective.

    z_ctx flows gradients; for ``jepa`` the target is detached (EMA output),
    for ``symalign``/``contrastive`` the raw target is used.
    """
    terms: dict = {}
    if objective == "jepa":
        assert predictor is not None, "jepa needs a predictor"
        # predict on the unit sphere: bounds the target magnitude so the loss
        # cannot diverge as VICReg spreads the (unnormalized) embeddings.
        pred = F.normalize(predictor(z_ctx), dim=-1)
        tgt = F.normalize(z_tgt.detach(), dim=-1)
        terms["pred"] = F.smooth_l1_loss(pred, tgt)
        v, c = vicreg_terms(z_ctx)
        terms["vic_var"], terms["vic_cov"] = v, c
        loss = terms["pred"] + vicreg_w[0] * v + vicreg_w[1] * c
    elif objective == "symalign":
        terms["align"] = F.mse_loss(z_ctx, z_tgt)
        loss = terms["align"]
    elif objective == "contrastive":
        terms["nce"] = info_nce(z_ctx, z_tgt, temperature)
        loss = terms["nce"]
    elif objective == "recon":
        loss = z_ctx.new_zeros(())
    else:
        raise ValueError(f"unknown objective: {objective}")
    scalar = {k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in terms.items()}
    scalar["rank_ctx"] = effective_rank(z_ctx)
    return loss, scalar
