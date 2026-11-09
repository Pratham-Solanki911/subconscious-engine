from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import SubconsciousEngineConfig, enforce_vram_ceiling
from .latent_bridge import LatentBridge
from .mmap_subconscious import MmapSubconsciousCore
from .ppo_router import DynamicEntropyRouter, RouterOutput


def compute_soft_superposition_embedding(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    mean_embedding_l2_norm: Optional[torch.Tensor] = None,
    top_k: int = 64,
    temperature: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training-Free Continuous Latent Superposition Step (Coconut / SoftCoT with L2 Norm Stabilization):

    1. Scales last-token logits at temperature `tau = 0.7`: `z = logits / tau` (in `float32`).
    2. Extracts Top-K (`K = 64`) logits and vocabulary indices: `z_top, idx_top = topk(z, 64)`.
    3. Computes probabilities in `float32`: `p = softmax(z_top)`.
    4. Forms the continuous superposition embedding vector against `W_E`:
           `e_raw = sum_{i=1}^K p_i * W_E[idx_top, i]`
    5. L2 Norm Stabilization: Scales `e_raw` in `float32` so its L2 norm matches the mean L2 norm
       of the model's embedding matrix `mu_{||W_E||_2}`, then casts back to `embedding_weight.dtype`.

    Args:
        logits: `[B, V]` or `[B, S, V]` output logits from the Active Mind.
        embedding_weight: `[V, d_small]` input token embedding matrix `W_E`.
        mean_embedding_l2_norm: Precomputed scalar `mu_{||W_E||_2}` in `float32`.
        top_k: Number of top token hypotheses (`K=64`).
        temperature: Softmax temperature (`T=0.7`).

    Returns:
        Tuple of `(e_latent, top_probs, top_indices)` where `e_latent` has shape `[B, 1, d_small]`.
    """
    last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
    work_dtype = embedding_weight.dtype

    # Compute probabilities in float32 for numerical stability
    scaled_logits_fp32 = last_logits.float() / max(temperature, 1e-4)
    k = min(top_k, scaled_logits_fp32.shape[-1])
    top_vals_fp32, top_indices = torch.topk(scaled_logits_fp32, k=k, dim=-1)  # [B, K]
    top_probs_fp32 = F.softmax(top_vals_fp32, dim=-1)  # [B, K]

    # Gather Top-K token embeddings from W_E: [B, K, d_small]
    selected_embeds_fp32 = embedding_weight[top_indices].float()

    # Weighted superposition in float32: [B, 1, K] @ [B, K, d_small] -> [B, 1, d_small]
    e_raw_fp32 = torch.bmm(top_probs_fp32.unsqueeze(1), selected_embeds_fp32)

    # L2 Norm Stabilization against mu_{||W_E||_2}
    if mean_embedding_l2_norm is None:
        with torch.no_grad():
            mean_embedding_l2_norm = embedding_weight.float().norm( p=2, dim=-1).mean()
    target_norm = mean_embedding_l2_norm.to(device=e_raw_fp32.device, dtype=torch.float32)
    current_norm = e_raw_fp32.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    e_stabilized_fp32 = e_raw_fp32 * (target_norm / current_norm)

    return e_stabilized_fp32.to(dtype=work_dtype), top_probs_fp32, top_indices


@dataclass
class GenerationTrace:
    """Detailed telemetry and output from an AnchoredSuperpositionEngine run."""

    mode: str
    prompt: str
    caveman_anchors_text: str
    final_answer: str
    full_response: str
    generated_token_count: int
    latent_steps_executed: int
    subconscious_triggers: int
    total_router_evaluations: int
    avg_active_subconscious_layers: float
    layer_entropy_history: List[float] = field(default_factory=list)
    latent_l2_norms: List[float] = field(default_factory=list)
    latency_sec: float = 0.0
    peak_vram_gb: float = 0.0
    raw_subconscious_ram_ms: float = 0.0
    effective_bridge_wait_ms: float = 0.0


def _execute_subconscious_worker(
    subconscious_core: MmapSubconsciousCore,
    h_up_cpu: torch.Tensor,
    mask_cpu: torch.Tensor,
) -> Tuple[torch.Tensor, int, float]:
    """Background CPU worker executing 7B System RAM middle layers in parallel with 1.5B GPU reasoning."""
    t_cpu0 = time.perf_counter()
    h_sub_cpu, executed_layers = subconscious_core(h_up_cpu, layer_mask=mask_cpu)
    raw_cpu_ms = (time.perf_counter() - t_cpu0) * 1000.0
    return h_sub_cpu, executed_layers, raw_cpu_ms


class AnchoredSuperpositionEngine(nn.Module):
    """
    Complete Disaggregated Subconscious-Active Latent Reasoning Engine.

    Integrates:
    - **Active Mind (GPU VRAM):** `Qwen/Qwen2.5-1.5B-Instruct` in `bfloat16`/`float16` (~3.1 GB).
    - **Forward Hook at Layer 14:** Connects `DynamicEntropyRouter` + `LatentBridge` + `MmapSubconsciousCore`.
    - **Caveman Symbolic Checkpoints + Soft-Token Superposition Loop:**
      Alternates between `caveman_anchor_tokens` discrete symbolic tokens (locking KV-cache geometry)
      and `latent_superposition_steps` continuous Top-K superposition embeddings (`inputs_embeds`),
      before decoding the final answer.
    """

    def __init__(
        self,
        config: SubconsciousEngineConfig,
        active_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        subconscious_core: Optional[MmapSubconsciousCore] = None,
        bridge: Optional[LatentBridge] = None,
        router: Optional[DynamicEntropyRouter] = None,
        load_subconscious: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        if torch.cuda.is_available() and config.gpu_device.startswith("cuda"):
            enforce_vram_ceiling(config.emulated_vram_ceiling_gb, device_idx=0)

        # 1. Load Active Mind on GPU in bfloat16/float16
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.active_model_id)
        else:
            self.tokenizer = tokenizer

        if active_model is None:
            self.active_model = AutoModelForCausalLM.from_pretrained(
                config.active_model_id,
                torch_dtype=config.dtype,
                device_map=config.gpu_device,
                attn_implementation="sdpa",
            )
        else:
            self.active_model = active_model

        self.active_model.eval()
        for p in self.active_model.parameters():
            p.requires_grad_(False)

        # Precompute mean L2 norm of the Active Mind's token embedding table W_E: mu_{||W_E||_2}
        # Compute in chunks so we never allocate a full [151936, 1536] float32 copy (892 MB) in VRAM
        with torch.no_grad():
            embed_weight = self.active_model.get_input_embeddings().weight
            chunk_norms = [
                chunk.float().norm(p=2, dim=-1)
                for chunk in torch.split(embed_weight, 16384, dim=0)
            ]
            self.register_buffer(
                "mean_embed_l2_norm",
                torch.cat(chunk_norms, dim=0).mean().detach(),
                persistent=False,
            )

        # 2. Initialize Trainable Latent Bridge (< 20 MB) & Dynamic PPO Router (< 5 MB) on GPU
        self.bridge = bridge if bridge is not None else LatentBridge(config)
        self.router = router if router is not None else DynamicEntropyRouter(config)

        # 3. Initialize Zero-Copy Memory-Mapped Subconscious Core (Layers 12..19 on CPU RAM)
        if subconscious_core is not None:
            self.subconscious_core: Optional[MmapSubconsciousCore] = subconscious_core
        elif load_subconscious:
            self.subconscious_core = MmapSubconsciousCore(config, auto_fetch=True)
        else:
            self.subconscious_core = None

        # Runtime hook state
        self._hook_enabled: bool = False
        self._hook_force_trigger: Optional[bool] = None
        self._hook_sample_policy: bool = False
        self._hook_effective_gate: Optional[float] = None
        self._hook_context_prefix_len: Optional[int] = None
        self._capture_layer14_prefill: bool = False
        self._async_launch_at_prefill: bool = False
        self._async_executor = ThreadPoolExecutor(max_workers=1)
        self._async_bridge_future: Optional[Future] = None
        self._layer14_history: Optional[torch.Tensor] = None
        self._cached_subconscious_delta: Optional[torch.Tensor] = None
        self._cached_subconscious_z_sub: Optional[torch.Tensor] = None
        self._cached_subconscious_gate: Optional[torch.Tensor] = None
        self._cached_subconscious_start_idx: Optional[int] = None
        self.last_router_output: Optional[RouterOutput] = None
        self.last_bridge_tensors: Optional[Dict[str, torch.Tensor]] = None
        self.telemetry_entropies: List[float] = []
        self.telemetry_active_layers: List[int] = []
        self.telemetry_triggers: int = 0
        self.telemetry_evals: int = 0
        self.last_raw_subconscious_ms: float = 0.0
        self.last_effective_wait_ms: float = 0.0

        # Attach forward hook at Layer 14 of Active Mind
        target_layer = self.active_model.model.layers[config.active_inject_layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._layer14_forward_hook)

    def reset_telemetry(self) -> None:
        self.last_router_output = None
        self.last_bridge_tensors = None
        self._hook_context_prefix_len = None
        self._capture_layer14_prefill = False
        self._async_launch_at_prefill = False
        self._async_bridge_future = None
        self._layer14_history = None
        self._cached_subconscious_delta = None
        self._cached_subconscious_z_sub = None
        self._cached_subconscious_gate = None
        self._cached_subconscious_start_idx = None
        self.telemetry_entropies.clear()
        self.telemetry_active_layers.clear()
        self.telemetry_triggers = 0
        self.telemetry_evals = 0
        self.last_raw_subconscious_ms = 0.0
        self.last_effective_wait_ms = 0.0

    def _layer14_forward_hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor | tuple):
        """
        Forward hook registered at Layer 14 of `Qwen/Qwen2.5-1.5B-Instruct`.
        Maintains the Layer-14 sequence hidden-state history across cached steps so
        `ContextFunnelCompressor` always attends over the full prompt + thinking trajectory,
        and caches the computed 7B Subconscious residual vector across the answer rollout.
        """
        is_tuple = isinstance(output, tuple)
        hidden_states: torch.Tensor = output[0] if is_tuple else output

        # Track Layer-14 prompt prefill hidden states so ContextFunnelCompressor sees the exact same
        # problem-statement manifold during both teacher-forced distillation and cached generation
        if self._capture_layer14_prefill:
            self._layer14_history = hidden_states.detach()
            if self._async_launch_at_prefill and self.subconscious_core is not None:
                router_out = self.router(
                    hidden_states=self._layer14_history,
                    final_norm=self.active_model.model.norm,
                    lm_head=self.active_model.lm_head,
                    sample=False,
                    force_trigger=self._hook_force_trigger,
                )
                self.last_router_output = router_out
                self.telemetry_evals += 1
                self.telemetry_entropies.append(float(router_out.entropy.mean().item()))
                if bool(router_out.should_trigger.any().item()):
                    self.telemetry_triggers += 1
                    h_funnel = self.bridge.compressor(self._layer14_history)
                    h_up = self.bridge.up_projector(h_funnel)
                    h_up_cpu = h_up.detach().to(device="cpu")
                    mask_cpu = router_out.layer_mask.detach().to(device="cpu")
                    self._async_bridge_future = self._async_executor.submit(
                        _execute_subconscious_worker,
                        self.subconscious_core,
                        h_up_cpu,
                        mask_cpu,
                    )

        if not self._hook_enabled:
            if self._cached_subconscious_z_sub is not None and self._cached_subconscious_gate is not None:
                z_sub = self._cached_subconscious_z_sub.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                gate = self._cached_subconscious_gate.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                if hidden_states.shape[1] > 1 and self._cached_subconscious_start_idx is not None:
                    # Inject Subconscious delta strictly onto the closing bracket token ']' of close_think_ids
                    # (when start_idx == 0) so KV-cache holds the 7B reasoning state at the thought boundary
                    # without perturbing the 'Concise Derivation & Final Answer:\n' formatting tokens or arithmetic.
                    if self._cached_subconscious_start_idx == 0:
                        first_tok = hidden_states[:, :1, :]
                        rest_toks = hidden_states[:, 1:, :]
                        tok_delta = self.bridge.down_projector.forward_token_conditioned(z_sub, first_tok)
                        updated = torch.cat([first_tok + (0.15 * gate) * tok_delta, rest_toks], dim=1)
                    else:
                        prefix_toks = hidden_states[:, : self._cached_subconscious_start_idx, :]
                        post_toks = hidden_states[:, self._cached_subconscious_start_idx :, :]
                        tok_delta = self.bridge.down_projector.forward_token_conditioned(z_sub, post_toks)
                        updated = torch.cat([prefix_toks, post_toks + gate * tok_delta], dim=1)
                    return (updated,) + output[1:] if is_tuple else updated
                return output
            if self._cached_subconscious_delta is not None:
                delta = self._cached_subconscious_delta.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                if hidden_states.shape[1] > 1 and self._cached_subconscious_start_idx is not None:
                    updated = hidden_states.clone()
                    updated[:, self._cached_subconscious_start_idx :, :] = (
                        updated[:, self._cached_subconscious_start_idx :, :] + delta
                    )
                else:
                    updated = hidden_states + delta
                return (updated,) + output[1:] if is_tuple else updated
            return output

        router_hidden = (
            hidden_states[:, : self._hook_context_prefix_len, :]
            if (self._hook_context_prefix_len is not None and 0 < self._hook_context_prefix_len <= hidden_states.shape[1])
            else hidden_states
        )
        router_out = self.router(
            hidden_states=router_hidden,
            final_norm=self.active_model.model.norm,
            lm_head=self.active_model.lm_head,
            sample=self._hook_sample_policy,
            force_trigger=self._hook_force_trigger,
        )
        self.last_router_output = router_out
        self.telemetry_evals += 1
        self.telemetry_entropies.append(float(router_out.entropy.mean().item()))

        if bool(router_out.should_trigger.any().item()):
            self.telemetry_triggers += 1
            updated_hidden, h_up, h_sub, executed_layers = self.bridge(
                hidden_states=hidden_states,
                subconscious_core=self.subconscious_core,
                layer_mask=router_out.layer_mask,
                force_effective_gate=self._hook_effective_gate,
                context_prefix_len=self._hook_context_prefix_len,
                sequence_history=self._layer14_history if (not self.training and self._hook_context_prefix_len is None) else None,
            )
            self.telemetry_active_layers.append(executed_layers)
            self.last_bridge_tensors = {
                "hidden_before": hidden_states,
                "hidden_after": updated_hidden,
                "h_up": h_up,
                "h_sub": h_sub,
            }
            if is_tuple:
                return (updated_hidden,) + output[1:]
            return updated_hidden

        self.telemetry_active_layers.append(0)
        return output

    @torch.no_grad()
    def generate_standard_cot(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> GenerationTrace:
        """Mode (a): Standard Autoregressive CoT on the Active Mind (no superposition, no RAM bridge)."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        self.reset_telemetry()
        self._hook_enabled = False

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise system design and formal logic solver. "
                    "Use ultra-compact equations and state the exact final answer immediately without filler."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.config.gpu_device)
        input_len = model_inputs.input_ids.shape[1]

        gen_ids = self.active_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = gen_ids[0, input_len:]
        answer_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - t0
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        )

        return GenerationTrace(
            mode="standard_cot",
            prompt=prompt,
            caveman_anchors_text="",
            final_answer=answer_text,
            full_response=answer_text,
            generated_token_count=int(new_ids.numel()),
            latent_steps_executed=0,
            subconscious_triggers=0,
            total_router_evaluations=0,
            avg_active_subconscious_layers=0.0,
            latency_sec=elapsed,
            peak_vram_gb=peak_vram,
            raw_subconscious_ram_ms=self.last_raw_subconscious_ms,
            effective_bridge_wait_ms=self.last_effective_wait_ms,
        )

    @torch.no_grad()
    def generate_anchored_superposition(
        self,
        prompt: str,
        enable_subconscious_bridge: bool = True,
        max_new_tokens: int = 256,
        num_cycles: Optional[int] = None,
        caveman_tokens_per_cycle: Optional[int] = None,
        latent_steps_per_cycle: Optional[int] = None,
        effective_bridge_gate: Optional[float] = None,
        async_overlap: Optional[bool] = None,
        sync_before_caveman: bool = False,
        preserve_superposition_kv: bool = False,
    ) -> GenerationTrace:
        """
        Execute the Hybrid Caveman Symbolic Checkpoints + Soft-Token Superposition Loop,
        optionally routed through the System RAM Subconscious Bridge when Layer 14 entropy > tau:

        1. Prefill the problem with a brief Caveman symbolic thinking header so discrete anchor tokens
           emit compressed state transitions (`[State -> Invariant -> Bound]`).
        2. Run `num_cycles` alternating cycles:
           - **Caveman Symbolic Checkpoint (`caveman_tokens_per_cycle` discrete tokens):**
             Locks KV-cache geometry and bounds continuous variance drift.
           - **Soft-Token Superposition (`latent_steps_per_cycle` continuous steps):**
             Computes `e_latent = sum_{i in Top-64} P(w_i) W_E[i]` at `T=0.7`, stabilizes its L2 norm
             to `mu_{||W_E||_2}`, and feeds `inputs_embeds=e_latent` directly into the next step.
        3. Inject the `</think>\nFinal Solution:\n` transition token and decode the final answer.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        self.reset_telemetry()

        cycles = num_cycles if num_cycles is not None else self.config.num_reasoning_cycles
        n_caveman = (
            caveman_tokens_per_cycle
            if caveman_tokens_per_cycle is not None
            else self.config.caveman_anchor_tokens
        )
        n_latent = (
            latent_steps_per_cycle
            if latent_steps_per_cycle is not None
            else self.config.latent_superposition_steps
        )

        self._hook_enabled = False
        self._hook_sample_policy = False
        self._hook_force_trigger = None
        learned_gate = float(torch.tanh(self.bridge.alpha).abs().item())
        self._hook_effective_gate = (
            effective_bridge_gate
            if effective_bridge_gate is not None
            else (None if learned_gate > 1e-4 else 0.015)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise system design and formal logic engine. "
                    "Think in compact symbolic arrows [Var -> Equation -> Result], then write a direct, concise derivation and state the exact final answer."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        think_prefix = base_prompt + "[Caveman Symbolic Trace: "
        inputs = self.tokenizer([think_prefix], return_tensors="pt").to(self.config.gpu_device)

        embed_layer = self.active_model.get_input_embeddings()
        embed_weight = embed_layer.weight

        use_async = (
            async_overlap
            if async_overlap is not None
            else self.config.async_gpu_cpu_overlap
        )

        # Initial prefill pass (keep prefill KV cache unperturbed; capture exact problem manifold at Layer 14)
        self._hook_enabled = False
        self._capture_layer14_prefill = True
        self._async_launch_at_prefill = bool(
            enable_subconscious_bridge and use_async and self.subconscious_core is not None
        )
        outputs = self.active_model(
            input_ids=inputs.input_ids,
            use_cache=True,
            return_dict=True,
        )
        self._capture_layer14_prefill = False
        self._async_launch_at_prefill = False
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]

        # Early Synchronization Option (Check 2): Synchronize right before the first dynamic Caveman token
        # so the Caveman Symbolic Trace itself is guided by the 7B Subconscious Bridge!
        if (
            sync_before_caveman
            and enable_subconscious_bridge
            and self.subconscious_core is not None
            and self._layer14_history is not None
        ):
            if self._async_bridge_future is not None:
                t_wait0 = time.perf_counter()
                h_sub_cpu, executed_layers, raw_cpu_ms = self._async_bridge_future.result()
                wait_ms = (time.perf_counter() - t_wait0) * 1000.0
                self._async_bridge_future = None
                self.last_raw_subconscious_ms = raw_cpu_ms
                self.last_effective_wait_ms = wait_ms
                self.telemetry_active_layers.append(executed_layers)
                h_sub = h_sub_cpu.to(
                    device=self._layer14_history.device,
                    dtype=self._layer14_history.dtype,
                )
                self._cached_subconscious_z_sub = self.bridge.down_projector.encode_subconscious(h_sub).detach()
                if self._hook_effective_gate is not None:
                    eff_gate = torch.tensor(self._hook_effective_gate, device=self._layer14_history.device)
                else:
                    eff_gate = torch.tanh(self.bridge.alpha)
                self._cached_subconscious_gate = eff_gate.detach()
                self._cached_subconscious_start_idx = int(inputs.input_ids.shape[1]) - 1
                # Refresh last_logits with the Subconscious residual injected at the last header token
                outputs = self.active_model(
                    input_ids=inputs.input_ids,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                last_logits = outputs.logits[:, -1, :]

        caveman_token_ids: List[int] = []
        latent_l2_norms: List[float] = []
        latent_steps_done = 0

        for _ in range(cycles):
            # Phase A: Discrete Caveman Symbolic Anchor Tokens (locks KV cache geometry on GPU)
            for _ in range(n_caveman):
                next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True)  # [1, 1]
                tok_int = int(next_token_id.item())
                if tok_int == self.tokenizer.eos_token_id:
                    break
                caveman_token_ids.append(tok_int)
                outputs = self.active_model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                last_logits = outputs.logits[:, -1, :]

            # Phase B: Continuous Soft-Token Superposition Steps (inputs_embeds = e_latent)
            for _ in range(n_latent):
                e_latent, _, _ = compute_soft_superposition_embedding(
                    logits=last_logits,
                    embedding_weight=embed_weight,
                    mean_embedding_l2_norm=self.mean_embed_l2_norm,
                    top_k=self.config.top_k_superposition,
                    temperature=self.config.superposition_temperature,
                )
                latent_l2_norms.append(float(e_latent.float().norm(p=2, dim=-1).mean().item()))
                outputs = self.active_model(
                    inputs_embeds=e_latent,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                last_logits = outputs.logits[:, -1, :]
                latent_steps_done += 1

        caveman_text = self.tokenizer.decode(caveman_token_ids, skip_special_tokens=True).strip()

        # Synchronization Point: If Subconscious Bridge is enabled, either await the async CPU worker
        # (launched at Layer 14 of prefill while GPU ran layers 15..27 + Caveman/Superposition cycles)
        # or run synchronously if async_overlap=False.
        if enable_subconscious_bridge and self.subconscious_core is not None and self._layer14_history is not None:
            if self._async_bridge_future is not None:
                t_wait0 = time.perf_counter()
                h_sub_cpu, executed_layers, raw_cpu_ms = self._async_bridge_future.result()
                wait_ms = (time.perf_counter() - t_wait0) * 1000.0
                self._async_bridge_future = None
                self.last_raw_subconscious_ms = raw_cpu_ms
                self.last_effective_wait_ms = wait_ms
                self.telemetry_active_layers.append(executed_layers)

                h_sub = h_sub_cpu.to(
                    device=self._layer14_history.device,
                    dtype=self._layer14_history.dtype,
                )
                self._cached_subconscious_z_sub = self.bridge.down_projector.encode_subconscious(h_sub).detach()
                if self._hook_effective_gate is not None:
                    eff_gate = torch.tensor(self._hook_effective_gate, device=self._layer14_history.device)
                else:
                    eff_gate = torch.tanh(self.bridge.alpha)
                self._cached_subconscious_gate = eff_gate.detach()
                self._cached_subconscious_start_idx = int(inputs.input_ids.shape[1])
            elif not use_async:
                router_out = self.router(
                    hidden_states=self._layer14_history,
                    final_norm=self.active_model.model.norm,
                    lm_head=self.active_model.lm_head,
                    sample=False,
                    force_trigger=self._hook_force_trigger,
                )
                self.last_router_output = router_out
                self.telemetry_evals += 1
                self.telemetry_entropies.append(float(router_out.entropy.mean().item()))
                if bool(router_out.should_trigger.any().item()):
                    self.telemetry_triggers += 1
                    t_sync0 = time.perf_counter()
                    updated_h, h_up, h_sub, executed_layers = self.bridge(
                        hidden_states=self._layer14_history,
                        subconscious_core=self.subconscious_core,
                        layer_mask=router_out.layer_mask,
                        force_effective_gate=self._hook_effective_gate,
                    )
                    sync_ms = (time.perf_counter() - t_sync0) * 1000.0
                    self.last_raw_subconscious_ms = sync_ms
                    self.last_effective_wait_ms = sync_ms
                    self.telemetry_active_layers.append(executed_layers)
                    self._cached_subconscious_z_sub = self.bridge.down_projector.encode_subconscious(h_sub).detach()
                    if self._hook_effective_gate is not None:
                        eff_gate = torch.tensor(self._hook_effective_gate, device=self._layer14_history.device)
                    else:
                        eff_gate = torch.tanh(self.bridge.alpha)
                    self._cached_subconscious_gate = eff_gate.detach()
                    self._cached_subconscious_start_idx = int(inputs.input_ids.shape[1])

            if preserve_superposition_kv:
                self._cached_subconscious_start_idx = 0
                close_think_ids = self.tokenizer.encode(
                    "]\nConcise Derivation & Final Answer:\n",
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(self.config.gpu_device)
                outputs = self.active_model(
                    input_ids=close_think_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            else:
                anchored_prompt = (
                    think_prefix + caveman_text + "]\nConcise Derivation & Final Answer:\n"
                )
                anchored_ids = self.tokenizer([anchored_prompt], return_tensors="pt").input_ids.to(
                    self.config.gpu_device
                )
                outputs = self.active_model(
                    input_ids=anchored_ids,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        else:
            # Close the latent thinking block and decode the final answer cleanly using the anchored context
            close_think_ids = self.tokenizer.encode(
                "]\nConcise Derivation & Final Answer:\n",
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.config.gpu_device)

            outputs = self.active_model(
                input_ids=close_think_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        answer_ids: List[int] = []
        for step_idx in range(max_new_tokens):
            tok_val = int(next_token.item())
            if tok_val == self.tokenizer.eos_token_id:
                break
            answer_ids.append(tok_val)
            outputs = self.active_model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        self._hook_enabled = False
        self._cached_subconscious_delta = None
        final_answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - t0
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        )
        avg_layers = (
            sum(self.telemetry_active_layers) / max(len(self.telemetry_active_layers), 1)
            if self.telemetry_active_layers
            else 0.0
        )

        return GenerationTrace(
            mode="subconscious_bridge" if enable_subconscious_bridge else "caveman_superposition",
            prompt=prompt,
            caveman_anchors_text=caveman_text,
            final_answer=final_answer,
            full_response=f"<think>[{caveman_text}] (+{latent_steps_done} latent steps)</think>\n{final_answer}",
            generated_token_count=len(caveman_token_ids) + len(answer_ids),
            latent_steps_executed=latent_steps_done,
            subconscious_triggers=self.telemetry_triggers,
            total_router_evaluations=self.telemetry_evals,
            avg_active_subconscious_layers=avg_layers,
            layer_entropy_history=list(self.telemetry_entropies),
            latent_l2_norms=latent_l2_norms,
            latency_sec=elapsed,
            peak_vram_gb=peak_vram,
            raw_subconscious_ram_ms=self.last_raw_subconscious_ms,
            effective_bridge_wait_ms=self.last_effective_wait_ms,
        )


if __name__ == "__main__":
    # Standalone tripwire check for Soft-Token Superposition & L2 Norm Stabilization
    torch.manual_seed(42)
    vocab_sz, d_model = 1000, 1536
    fake_we = torch.randn(vocab_sz, d_model, dtype=torch.bfloat16) * 0.04
    expected_norm = fake_we.float().norm(p=2, dim=-1).mean()
    fake_logits = torch.randn(2, vocab_sz, dtype=torch.bfloat16) * 3.0
    e_lat, probs, idx = compute_soft_superposition_embedding(
        fake_logits, fake_we, mean_embedding_l2_norm=expected_norm, top_k=64, temperature=0.7
    )
    actual_norm = e_lat.float().norm(p=2, dim=-1).mean()
    assert e_lat.shape == (2, 1, d_model)
    assert probs.shape == (2, 64)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)
    assert torch.allclose(actual_norm, expected_norm, rtol=1e-2)
    print(
        f"anchored_superposition.py tripwire: ok (target_L2={expected_norm:.4f}, actual_L2={actual_norm:.4f})"
    )
