"""
Disaggregated Subconscious-Active Latent Reasoning Engine.

Modules:
- config: Architectural, hardware, and training hyperparameters.
- mmap_subconscious: Zero-copy safetensors shard loader for isolated CPU middle layers.
- latent_bridge: Trainable Attention-Funnel, W_up, RMSNorm, and W_down residual bridge.
- ppo_router: Entropy-triggered RL policy for dynamic layer masking + log_probs.
- anchored_superposition: Caveman 3-token anchors + Soft-Token Superposition loop.
- trainer: Teacher-forced distillation & PPO training loop for Bridge + Router.
"""

from .config import SubconsciousEngineConfig
from .mmap_subconscious import MmapSubconsciousCore
from .latent_bridge import LatentBridge
from .ppo_router import DynamicEntropyRouter, RouterOutput
from .anchored_superposition import AnchoredSuperpositionEngine, GenerationTrace
from .trainer import SubconsciousTrainer

__all__ = [
    "SubconsciousEngineConfig",
    "MmapSubconsciousCore",
    "LatentBridge",
    "DynamicEntropyRouter",
    "RouterOutput",
    "AnchoredSuperpositionEngine",
    "GenerationTrace",
    "SubconsciousTrainer",
]
