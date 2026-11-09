"""Compatibility alias exposing LatentBridge, ContextFunnelCompressor, and MmapSubconsciousCore."""
from .latent_bridge import ContextFunnelCompressor, DownProjector, LatentBridge, RMSNorm, UpProjector
from .mmap_subconscious import MmapSubconsciousCore, ensure_surgical_subconscious_shards

__all__ = [
    "ContextFunnelCompressor",
    "UpProjector",
    "DownProjector",
    "RMSNorm",
    "LatentBridge",
    "MmapSubconsciousCore",
    "ensure_surgical_subconscious_shards",
]
