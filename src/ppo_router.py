from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .config import SubconsciousEngineConfig


@dataclass
class RouterOutput:
    """Structured output from the Dynamic PPO Entropy Router."""

    entropy: torch.Tensor  # [B] Shannon entropy in nats: -sum p_i log(p_i + 1e-9)
    should_trigger: torch.Tensor  # [B] bool tensor: entropy > tau_entropy
    layer_mask: torch.Tensor  # [B, L] binary mask in {0, 1}^L (L=8 for layers 12..19)
    log_probs: torch.Tensor  # [B] summed Bernoulli log-probabilities log pi_theta(a | h_t, H)
    per_layer_probs: torch.Tensor  # [B, L] Bernoulli probabilities sigma(logits)
    value_estimate: torch.Tensor  # [B] critic value baseline V(h_t, H) for PPO advantage
    policy_entropy: torch.Tensor  # [B] Bernoulli policy entropy for exploration bonus


class DynamicEntropyRouter(nn.Module):
    """
    Dynamic PPO / RL Entropy Router across PCIe (`< 5 MB` in GPU VRAM).

    1. Computes early Shannon entropy at Layer 14 using the Active Mind's final norm and `lm_head`
       (or directly from provided intermediate logits) in `float32`:
           `H(p) = -sum_i p_i * log(p_i + 1e-9)`
    2. If `H(p) > tau_entropy` (default `2.2` nats), a 2-layer MLP policy `pi_theta(a | h_t, H)`
       samples (or thresholds during eval) a binary layer execution mask `a in {0, 1}^L ~ Bernoulli(logits)`.
    3. Computes the exact reward and PPO clipped surrogate loss:
           `R = Delta LogLikelihood(y_gold) - lambda_1 * sum_{l=1}^L a_l - lambda_2 * I(PCIe_Triggered)`
    """

    def __init__(self, config: SubconsciousEngineConfig) -> None:
        super().__init__()
        self.config = config
        self.d_small = config.d_small
        self.num_layers = config.num_subconscious_layers  # L = 8 for layers [12..19]
        self.hidden_dim = config.router_hidden_dim
        self.tau_entropy = config.entropy_threshold

        # Input features: last-token hidden state h_t [d_small] + scalar entropy H(p) + normalized H(p)
        in_features = self.d_small + 2
        self.input_norm = nn.LayerNorm(self.d_small)
        self.fc1 = nn.Linear(in_features, self.hidden_dim)
        self.act = nn.SiLU()
        self.policy_head = nn.Linear(self.hidden_dim, self.num_layers)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        # Bias policy slightly positive (+0.85 -> sigmoid ~ 0.70) so ~5-6 of 8 layers start active when triggered
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.policy_head.weight, gain=0.5)
        nn.init.constant_(self.policy_head.bias, 0.85)
        nn.init.xavier_uniform_(self.value_head.weight, gain=0.5)
        nn.init.zeros_(self.value_head.bias)

        self.to(device=config.gpu_device, dtype=config.dtype)

    def parameter_megabytes(self) -> float:
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        return total_bytes / (1024 ** 2)

    @staticmethod
    def compute_shannon_entropy(logits: torch.Tensor, top_m: int = 256) -> torch.Tensor:
        """
        Compute Shannon entropy in nats `H(p) = -sum p_i log(p_i + 1e-9)` in `float32`.

        Args:
            logits: `[B, V]` or `[B, S, V]` vocabulary logits.
            top_m: If `0 < top_m < V`, evaluates over the top-M logits (captures >99.9% probability mass
                   while avoiding full 152K-vocab fp32 allocation). Set `top_m=0` for full vocabulary.
        Returns:
            Tensor of shape `[B]` in `float32` representing Shannon entropy in nats.
        """
        last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        logits_fp32 = last_logits.float()
        if 0 < top_m < logits_fp32.shape[-1]:
            logits_fp32, _ = torch.topk(logits_fp32, k=top_m, dim=-1)
        probs = F.softmax(logits_fp32, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy

    @torch.no_grad()
    def compute_early_layer_entropy(
        self,
        hidden_states: torch.Tensor,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ) -> torch.Tensor:
        """
        Compute early Shannon entropy at Layer 14 using the Active Mind's final norm and `lm_head`
        under `torch.no_grad()`.
        """
        last_hidden = hidden_states[:, -1:, :]  # [B, 1, d_small]
        normed = final_norm(last_hidden)
        early_logits = lm_head(normed)  # [B, 1, V]
        return self.compute_shannon_entropy(early_logits, top_m=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        final_norm: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
        sample: bool = True,
        force_trigger: Optional[bool] = None,
        threshold_override: Optional[float] = None,
    ) -> RouterOutput:
        """
        Evaluate the Dynamic PPO Router on `hidden_states` `[B, S, d_small]`.

        Args:
            hidden_states: Active Mind Layer 14 hidden states `[B, S, d_small]`.
            entropy: Precomputed `[B]` Shannon entropy tensor, or computed via `final_norm` + `lm_head`.
            final_norm: Active Mind's `model.norm` (used if `entropy is None`).
            lm_head: Active Mind's `lm_head` (used if `entropy is None`).
            sample: If True, samples `a ~ Bernoulli(logits)`; if False, uses deterministic `(probs >= 0.5)`.
            force_trigger: Optional override for testing/ablation (`True` = always trigger, `False` = never).
            threshold_override: Optional custom entropy threshold `tau`.
        """
        bsz = hidden_states.shape[0]
        device = hidden_states.device
        work_dtype = self.fc1.weight.dtype

        if entropy is None:
            if final_norm is not None and lm_head is not None:
                entropy = self.compute_early_layer_entropy(hidden_states, final_norm, lm_head)
            else:
                # Fallback token variance proxy if neither entropy nor lm_head is supplied
                entropy = hidden_states[:, -1, :].float().std(dim=-1) * 2.5

        entropy_fp32 = entropy.to(device=device, dtype=torch.float32).view(bsz)
        tau = self.tau_entropy if threshold_override is None else threshold_override

        if force_trigger is True:
            should_trigger = torch.ones(bsz, dtype=torch.bool, device=device)
        elif force_trigger is False:
            should_trigger = torch.zeros(bsz, dtype=torch.bool, device=device)
        else:
            should_trigger = entropy_fp32 > tau

        # Build policy input from last token hidden state + entropy statistics
        h_last = self.input_norm(hidden_states[:, -1, :].to(dtype=work_dtype))
        ent_feat = torch.stack([entropy_fp32, (entropy_fp32 - tau) / max(tau, 1e-4)], dim=-1).to(dtype=work_dtype)
        router_in = torch.cat([h_last, ent_feat], dim=-1)  # [B, d_small + 2]

        hidden = self.act(self.fc1(router_in))
        logits_fp32 = self.policy_head(hidden).float()  # [B, L]
        value_fp32 = self.value_head(hidden).float().squeeze(-1)  # [B]

        dist = Bernoulli(logits=logits_fp32)
        probs_fp32 = dist.probs  # [B, L]

        if sample:
            raw_mask = dist.sample()  # [B, L] in {0.0, 1.0}
        else:
            raw_mask = (probs_fp32 >= 0.5).float()

        # Guarantee at least 1 Subconscious layer runs when triggered so a trigger is never a no-op
        all_zero = raw_mask.sum(dim=-1) == 0
        if all_zero.any():
            top_layer_idx = probs_fp32.argmax(dim=-1, keepdim=True)
            fallback_one_hot = torch.zeros_like(raw_mask).scatter_(1, top_layer_idx, 1.0)
            raw_mask = torch.where(all_zero.unsqueeze(-1), fallback_one_hot, raw_mask)

        # Zero out the execution mask for batch elements that did NOT exceed the entropy threshold
        trigger_float = should_trigger.float().unsqueeze(-1)
        final_mask = raw_mask * trigger_float  # [B, L]

        log_probs = dist.log_prob(raw_mask).sum(dim=-1)  # [B]
        policy_entropy = dist.entropy().sum(dim=-1)  # [B]

        return RouterOutput(
            entropy=entropy_fp32,
            should_trigger=should_trigger,
            layer_mask=final_mask,
            log_probs=log_probs,
            per_layer_probs=probs_fp32,
            value_estimate=value_fp32,
            policy_entropy=policy_entropy,
        )

    def evaluate_actions(
        self,
        hidden_states: torch.Tensor,
        entropy: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Re-evaluate log_probs, values, and policy entropy for stored `(hidden_states, entropy, actions)` in PPO."""
        bsz = hidden_states.shape[0]
        work_dtype = self.fc1.weight.dtype
        tau = self.tau_entropy
        entropy_fp32 = entropy.float().view(bsz)

        h_last = self.input_norm(hidden_states[:, -1, :].to(dtype=work_dtype))
        ent_feat = torch.stack([entropy_fp32, (entropy_fp32 - tau) / max(tau, 1e-4)], dim=-1).to(dtype=work_dtype)
        router_in = torch.cat([h_last, ent_feat], dim=-1)

        hidden = self.act(self.fc1(router_in))
        logits_fp32 = self.policy_head(hidden).float()
        value_fp32 = self.value_head(hidden).float().squeeze(-1)

        dist = Bernoulli(logits=logits_fp32)
        log_probs = dist.log_prob(actions.float()).sum(dim=-1)
        policy_entropy = dist.entropy().sum(dim=-1)
        return {
            "log_probs": log_probs,
            "values": value_fp32,
            "policy_entropy": policy_entropy,
        }

    def compute_reward(
        self,
        delta_log_likelihood: torch.Tensor,
        layer_mask: torch.Tensor,
        pcie_triggered: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the exact RL reward:
            `R = Delta LogLikelihood(y_gold) - lambda_1 * sum_{l=1}^8 a_l - lambda_2 * I(PCIe_Triggered)`
        """
        active_layers = layer_mask.float().sum(dim=-1)
        triggered_indicator = pcie_triggered.float()
        reward = (
            delta_log_likelihood.float()
            - self.config.layer_cost_penalty_lambda1 * active_layers
            - self.config.pcie_trigger_penalty_lambda2 * triggered_indicator
        )
        return reward

    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        policy_entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO clipped surrogate policy + critic value + entropy bonus loss."""
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        adv = advantages.detach()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip_eps, 1.0 + self.config.ppo_clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns.detach())
        entropy_loss = -policy_entropy.mean()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + self.config.entropy_bonus_coef * entropy_loss
        )
        metrics = {
            "ppo_total_loss": float(total_loss.item()),
            "ppo_policy_loss": float(policy_loss.item()),
            "ppo_value_loss": float(value_loss.item()),
            "ppo_entropy": float(policy_entropy.mean().item()),
        }
        return total_loss, metrics


if __name__ == "__main__":
    cfg = SubconsciousEngineConfig(gpu_device="cpu")
    router = DynamicEntropyRouter(cfg)
    mb = router.parameter_megabytes()
    assert mb < 5.0, f"Router exceeded 5 MB ceiling: {mb:.2f} MB"
    h = torch.randn(2, 6, cfg.d_small, dtype=cfg.dtype)
    fake_ent = torch.tensor([1.5, 3.1])  # sample 0 below tau=2.2, sample 1 above tau=2.2
    out = router(h, entropy=fake_ent, sample=True)
    assert out.should_trigger.tolist() == [False, True]
    assert out.layer_mask[0].sum().item() == 0.0
    assert out.layer_mask[1].sum().item() >= 1.0
    assert out.log_probs.shape == (2,)
    print(f"ppo_router.py tripwire: ok (VRAM={mb:.2f} MB)")
