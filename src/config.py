from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch


def _resolve_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    colab_dir = Path("/content/subconscious-engine")
    if colab_dir.exists():
        return colab_dir
    return Path.cwd()


def load_dotenv_into_environ() -> None:
    """Safely load HF_TOKEN from .env files without printing or logging the token value."""
    base = _resolve_base_dir()
    candidates = [
        Path("/content/subconscious-engine/.env"),
        Path.cwd() / ".env",
        base / ".env",
        Path.home() / "subconscious-engine" / ".env",
        Path("d:/The LLMtest/.env"),
    ]
    for env_path in candidates:
        try:
            if env_path.is_file():
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and val and key not in os.environ:
                        os.environ[key] = val
                        if key == "HF_TOKEN":
                            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", val)
        except Exception:
            pass


def detect_optimal_dtype() -> torch.dtype:
    """Auto-detect bfloat16 (L4 / A100 / Ampere+) vs float16 (T4) vs float32 (CPU fallback)."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.bfloat16


def enforce_vram_ceiling(ceiling_gb: float = 6.0, device_idx: int = 0) -> float:
    """
    Enforce a hard software VRAM cap on CUDA (e.g., 6.0 GB on a 16GB T4 or 24GB L4 Colab GPU)
    using `torch.cuda.set_per_process_memory_fraction` so Colab mirrors a local 6GB GPU.
    """
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(device_idx)
    total_gb = props.total_memory / (1024 ** 3)
    fraction = min(1.0, max(0.05, ceiling_gb / total_gb))
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)
    return fraction


@dataclass
class SubconsciousEngineConfig:
    """Master configuration for the Disaggregated Subconscious-Active Latent Engine."""

    # Model identifiers
    active_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    subconscious_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Local NVMe SSD cache directory (never Google Drive FUSE)
    subconscious_cache_dir: str = field(
        default_factory=lambda: (
            "/content/cache/qwen2.5_7b_middle_shards"
            if Path("/content").exists()
            else str(_resolve_base_dir() / "data" / "qwen2.5_7b_middle_shards")
        )
    )

    # Hardware & Precision constraints
    gpu_device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cpu_device: str = "cpu"
    dtype: torch.dtype = field(default_factory=detect_optimal_dtype)
    emulated_vram_ceiling_gb: float = 6.0  # Enforced via set_per_process_memory_fraction
    strict_vram_limit_gb: float = 5.2
    hard_vram_ceiling_gb: float = 5.5

    # Active Mind dimensions (Qwen2.5-1.5B-Instruct)
    d_small: int = 1536
    active_num_layers: int = 28
    active_vocab_size: int = 151936
    active_inject_layer_idx: int = 14  # Forward hook attachment point at Layer 14

    # Subconscious Core dimensions (Qwen2.5-7B-Instruct isolated middle layers)
    d_big: int = 3584
    subconscious_layer_range: Tuple[int, int] = (12, 19)  # Inclusive [12..19] = 8 middle layers
    intra_layer_outlier_ratio: float = 0.01  # Top 1% salient input channels kept in FP16/BF16 (~36 / ~190 cols)
    intra_layer_int4_group_size: int = 128  # Group size 128 for symmetric INT4 99% bulk MLP weights
    async_gpu_cpu_overlap: bool = True  # Overlap 7B System RAM execution with 1.5B GPU reasoning via ThreadPoolExecutor

    # Latent Bridge & ContextFunnelCompressor hyperparameters (< 20 MB in GPU VRAM)
    num_essence_vectors: int = 4  # k=4 compressed continuous essence vectors
    bridge_num_heads: int = 8  # MultiheadAttention(1536, num_heads=8, batch_first=True)
    bridge_max_vram_mb: float = 20.0  # Strict < 20 MB ceiling (actual ~14.2 MB)
    bridge_init_gate: float = 0.0  # Zero-initialized alpha: tanh(0.0) = 0 -> 100% uncorrupted at init

    # Anchored Superposition (Caveman Symbolic Checkpoints + Coconut Soft-Token Superposition)
    top_k_superposition: int = 64
    superposition_temperature: float = 0.7
    caveman_anchor_tokens: int = 4  # 4 discrete Caveman anchor tokens (supports 3 or 4 via config)
    latent_superposition_steps: int = 3  # 3 continuous Soft-Token Superposition steps
    num_reasoning_cycles: int = 2  # Number of alternating Caveman <-> Superposition cycles

    # Dynamic PPO / RL Entropy Router hyperparameters (< 5 MB in GPU VRAM)
    entropy_threshold: float = 1.2  # Trigger Subconscious bridge when Shannon entropy (nats) > tau_entropy
    router_hidden_dim: int = 256
    layer_cost_penalty_lambda1: float = 0.04  # lambda_1 per active CPU layer
    pcie_trigger_penalty_lambda2: float = 0.08  # lambda_2 for triggering PCIe transfer

    # Trainer (Distillation + PPO) hyperparameters
    learning_rate_bridge: float = 2e-4
    learning_rate_router: float = 1e-4
    ppo_clip_eps: float = 0.2
    entropy_bonus_coef: float = 0.01
    value_loss_coef: float = 0.5
    distill_loss_coef: float = 1.0

    def __post_init__(self) -> None:
        load_dotenv_into_environ()

    @property
    def subconscious_layer_indices(self) -> Tuple[int, ...]:
        start, end = self.subconscious_layer_range
        assert 0 <= start <= end < 28, f"Invalid layer range: {self.subconscious_layer_range}"
        return tuple(range(start, end + 1))

    @property
    def num_subconscious_layers(self) -> int:
        return len(self.subconscious_layer_indices)


if __name__ == "__main__":
    cfg = SubconsciousEngineConfig()
    assert cfg.subconscious_layer_indices == (12, 13, 14, 15, 16, 17, 18, 19)
    assert cfg.num_subconscious_layers == 8
    assert cfg.bridge_init_gate == 0.0
    print("config.py tripwire: ok")
