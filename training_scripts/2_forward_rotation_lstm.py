from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


motion = load_module(
    "nips_forward_rotation_lstm_motion_base",
    Path(__file__).with_name("2_forward_rotation_gnn.py"),
)

base = motion.base
device = motion.device
torch.set_default_dtype(torch.float32)

LATENT_DIM = motion.LATENT_DIM
MODEL_DIM = motion.MODEL_DIM
DROPOUT = motion.DROPOUT
EPOCHS = motion.EPOCHS
TRAIN_BATCH_SIZE = motion.TRAIN_BATCH_SIZE
LR = motion.LR
WEIGHT_DECAY = motion.WEIGHT_DECAY
VAE_WEIGHTS = motion.VAE_WEIGHTS
OUT_ROOT = motion.OUT_ROOT
RNN_LAYERS = 2
EARLY_STOP_PATIENCE = motion.EARLY_STOP_PATIENCE
EARLY_STOP_MIN_DELTA = motion.EARLY_STOP_MIN_DELTA
ALIAS = "0507_lstm"

ImgOnlyEncoder = motion.ImgOnlyEncoder
strip_module = motion.strip_module
encode_all = motion.encode_all
build_sequences = motion.build_sequences
collate = motion.collate
fixed_frame_dir = motion.fixed_frame_dir
rollout_planar = motion.rollout_planar
identity_planar_rotation = motion.identity_planar_rotation
planar_rotation_target = motion.planar_rotation_target
loss_terms = motion.loss_terms
run_epoch = motion.run_epoch


class ActionLSTM(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = MODEL_DIM,
        rnn_layers: int = RNN_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = torch.zeros(latents.shape[:2], device=latents.device, dtype=torch.bool)

        batch_size = latents.size(0)
        x = self.encoder(latents).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        h, _ = self.rnn(x)
        decoded = self.decoder(h)
        moves = decoded[..., :2].masked_fill(padding_mask.unsqueeze(-1), 0.0)
        rotations = decoded[..., 2:].masked_fill(padding_mask.unsqueeze(-1), 0.0)
        moves[:, 0] = 0.0
        rotations[:, 0] = 0.0
        locs, dirs = rollout_planar(
            latents.new_zeros(batch_size, 3),
            fixed_frame_dir(latents.new_zeros(batch_size, 3)),
            moves[:, 1:],
            rotations[:, 1:],
        )
        locs = locs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        dirs = dirs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return dict(loc=locs, direction=dirs, move=moves, rotation=rotations)


ActionGNN = ActionLSTM


def train():
    out = OUT_ROOT / ALIAS
    encoder = base.load_encoder()
    sequences = build_sequences(encode_all(encoder))
    train_sequences, val_sequences = base.split_sequences(sequences)
    train_loader = DataLoader(base.Walks(train_sequences), batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(base.Walks(val_sequences), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)
    model = ActionLSTM().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS)

    base.write_split(out / "split.csv", train_sequences, val_sequences)
    print(
        f"training {ALIAS}: {len(train_sequences)} train, {len(val_sequences)} val sequences, "
        f"rnn_layers={RNN_LAYERS}"
    )

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(EPOCHS):
        train_losses = run_epoch(model, train_loader, optim)
        val_losses = run_epoch(model, val_loader)
        sched.step()

        improved = val_losses["loss"] == val_losses["loss"] and val_losses["loss"] < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val = val_losses["loss"]
            best_epoch = epoch
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model_weights_best.pt")

        row = {
            "epoch": epoch,
            **base.prefix_keys("train", train_losses),
            **base.prefix_keys("val", val_losses),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "is_best": int(improved),
            "num_sequences": len(sequences),
            "num_train_sequences": len(train_sequences),
            "num_val_sequences": len(val_sequences),
            "rnn_layers": RNN_LAYERS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        }
        base.append_csv(out / "losses.csv", row)
        if epoch % 10 == 0:
            print(row)
        if epoch % 1000 == 0:
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / f"model_weights_{epoch}.pt")
        if best_epoch >= 0 and epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(
                f"early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}, best_val_loss={best_val}"
            )
            break

    torch.save(model.state_dict(), out / "model_weights.pt")


if __name__ == "__main__":
    train()
