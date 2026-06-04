from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


depthprior = load_module(
    "nips_transform_gnn_anglediff_depthprior_base",
    Path(__file__).with_name("2_transform_gnn_anglediff_depthprior.py"),
)

ALIAS = "0521_gnn_k2_l2_d64_anglediff_depthprior_sceneholdout"
SCENE_VAL_FRACTION = 0.2
HELDOUT_SCENES: Tuple[str, ...] = ()


def split_sequences_by_scene(sequences: Sequence) -> Tuple[list, list]:
    by_scene: Dict[str, list] = {}
    for sequence in sequences:
        by_scene.setdefault(sequence.scene_name, []).append(sequence)

    scenes = sorted(by_scene)
    if len(scenes) < 2:
        return list(sequences), []

    if HELDOUT_SCENES:
        val_scenes = set(HELDOUT_SCENES)
    else:
        generator = torch.Generator().manual_seed(depthprior.SPLIT_SEED)
        order = torch.randperm(len(scenes), generator=generator).tolist()
        num_val = min(len(scenes) - 1, max(1, round(len(scenes) * SCENE_VAL_FRACTION)))
        val_scenes = {scenes[idx] for idx in order[:num_val]}

    train, val = [], []
    for scene in scenes:
        target = val if scene in val_scenes else train
        target.extend(by_scene[scene])

    if not train:
        raise ValueError(f"Scene holdout left no training scenes. HELDOUT_SCENES={sorted(val_scenes)}")
    if not val:
        raise ValueError(f"Scene holdout left no validation scenes. HELDOUT_SCENES={sorted(val_scenes)}")

    print(f"scene-holdout validation scenes: {sorted(val_scenes)}")
    return train, val


depthprior.ALIAS = ALIAS
depthprior.split_sequences = split_sequences_by_scene


def train() -> None:
    depthprior.train()


if __name__ == "__main__":
    train()
