"""Cost-triple: wall-clock latency, peak GPU memory, FLOPs/params.

Torch is optional: importing this module never requires torch, so the pure
pose/calibration metrics stay usable on a laptop without a DL stack. The
GPU/FLOPs helpers import torch lazily and degrade gracefully.
"""
from __future__ import annotations

import time
from typing import Callable, Optional


def count_params(model) -> int:
    """Total number of parameters of a torch.nn.Module."""
    return int(sum(p.numel() for p in model.parameters()))


def measure_latency(fn: Callable[[], object], warmup: int = 5, iters: int = 50) -> float:
    """Mean wall-clock seconds per call of ``fn`` (CUDA-synced if available)."""
    sync = _cuda_sync()
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / max(iters, 1)


def reset_peak_gpu_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_gpu_mem_mb() -> Optional[float]:
    """Peak allocated CUDA memory in MB, or None if CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


def flops(model, example_inputs) -> Optional[float]:
    """Best-effort forward FLOPs (= 2 x MACs). Both backends return the SAME
    convention so the value is comparable across heterogeneous nodes.
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        # fvcore's .total() counts MACs (despite the name); double for FLOPs.
        return float(FlopCountAnalysis(model, example_inputs).total()) * 2.0
    except Exception:
        pass
    try:
        from ptflops import get_model_complexity_info  # type: ignore

        macs, _ = get_model_complexity_info(
            model, tuple(example_inputs.shape[1:]), as_strings=False,
            print_per_layer_stat=False,
        )
        return float(macs) * 2.0  # MACs -> FLOPs
    except Exception:
        return None


def cost_triple(model, forward_fn, example_inputs=None,
                warmup: int = 5, iters: int = 50) -> dict:
    """Bundle: latency_s, peak_mem_mb, flops, params.

    ``forward_fn`` is a no-arg callable running one forward pass;
    ``example_inputs`` (optional) is passed to the FLOPs counter.
    """
    reset_peak_gpu_mem()
    latency = measure_latency(forward_fn, warmup=warmup, iters=iters)
    return {
        "latency_s": latency,
        "peak_mem_mb": peak_gpu_mem_mb(),
        "flops": flops(model, example_inputs) if example_inputs is not None else None,
        "params": count_params(model),
    }


def _cuda_sync() -> Callable[[], None]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:
        pass
    return lambda: None
