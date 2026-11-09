"""Compatibility alias exposing Soft-Token Superposition & Caveman Anchored Engine."""
from .anchored_superposition import (
    AnchoredSuperpositionEngine,
    GenerationTrace,
    compute_soft_superposition_embedding,
)

__all__ = [
    "AnchoredSuperpositionEngine",
    "GenerationTrace",
    "compute_soft_superposition_embedding",
]
