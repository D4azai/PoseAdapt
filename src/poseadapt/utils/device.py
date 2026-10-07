"""Compute-device selection with lazy PyTorch imports."""

from __future__ import annotations


def resolve_device(requested: str | None = "auto", *, require_torch: bool = False) -> str:
    """Resolve `auto` to CUDA when available, otherwise CPU.

    Explicit CUDA requests fail early when CUDA is unavailable. PyTorch remains an
    optional dependency for core utilities and documentation commands.
    """
    normalized = (requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"} and not normalized.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")

    try:
        import torch
    except ImportError as error:
        if require_torch or normalized != "cpu":
            raise RuntimeError("PyTorch is required for model device selection") from error
        return "cpu"

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable")
    return normalized
