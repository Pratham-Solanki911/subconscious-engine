from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psutil
import torch

# Ensure project root is on sys.path (works both in standard Python and Colab Jupyter kernel)
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else (Path("/content/subconscious-engine") if Path("/content/subconscious-engine").exists() else Path.cwd())
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anchored_superposition import AnchoredSuperpositionEngine, compute_soft_superposition_embedding
from src.config import SubconsciousEngineConfig, enforce_vram_ceiling
from src.trainer import SubconsciousTrainer


def run_verification_pipeline() -> dict:
    print("=" * 80)
    print("DISAGGREGATED SUBCONSCIOUS-ACTIVE LATENT ENGINE - VERIFICATION & VRAM AUDIT")
    print("=" * 80)

    cfg = SubconsciousEngineConfig()
    if torch.cuda.is_available():
        frac = enforce_vram_ceiling(cfg.emulated_vram_ceiling_gb, device_idx=0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"[Hardware] GPU: {gpu_name} ({total_vram_gb:.2f} GB physical, "
            f"capped at {cfg.emulated_vram_ceiling_gb:.2f} GB via fraction={frac:.3f}) | Dtype: {cfg.dtype}"
        )
    else:
        print("[Hardware] WARNING: CUDA not available, running on CPU fallback.")

    vm = psutil.virtual_memory()
    print(f"[Hardware] System RAM: {vm.total / (1024**3):.2f} GB total, {vm.available / (1024**3):.2f} GB free")

    # 1. Initialize Complete Engine (Active Mind on GPU + Mmap Subconscious Layers [12..19] on CPU RAM)
    t0 = time.perf_counter()
    engine = AnchoredSuperpositionEngine(cfg, load_subconscious=True)
    init_sec = time.perf_counter() - t0

    # 2. Verify MmapSubconsciousCore Isolation & Zero-CUDA Guarantee
    assert engine.subconscious_core is not None
    sub_audit = engine.subconscious_core.verify_zero_cuda_and_isolation()
    print(f"[Subconscious Mmap Audit] {json.dumps(sub_audit)}")
    assert sub_audit["all_cpu"] is True
    assert sub_audit["num_layers"] == 8  # Layers [12..19]

    # 3. Verify Bridge (< 20 MB) & Router (< 5 MB) VRAM Footprints & Zero-Initialized Gate
    bridge_mb = engine.bridge.parameter_megabytes()
    router_mb = engine.router.parameter_megabytes()
    init_alpha = float(engine.bridge.alpha.item())
    print(
        f"[Module Footprints] LatentBridge: {bridge_mb:.2f} MB (< 20 MB limit) | "
        f"DynamicEntropyRouter: {router_mb:.2f} MB (< 5 MB limit) | init alpha: {init_alpha:.4f}"
    )
    assert bridge_mb < 20.0, f"Bridge exceeded 20 MB limit: {bridge_mb:.2f} MB"
    assert router_mb < 5.0, f"Router exceeded 5 MB limit: {router_mb:.2f} MB"
    assert abs(init_alpha) < 1e-7, f"Expected alpha=0.0 at init, got {init_alpha}"

    # Verify exact zero-corruption of residual stream at initialization (alpha == 0.0)
    dummy_h = torch.randn(1, 8, cfg.d_small, device=cfg.gpu_device, dtype=cfg.dtype)
    with torch.no_grad():
        h_inj_init, h_up_init, h_sub_init, n_exec = engine.bridge(
            dummy_h,
            subconscious_core=engine.subconscious_core,
            layer_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], device=cfg.gpu_device),
        )
    max_init_diff = float((h_inj_init - dummy_h).abs().max().item())
    print(
        f"[Zero-Gate Invariant] H_up shape={tuple(h_up_init.shape)}, H_sub shape={tuple(h_sub_init.shape)}, "
        f"executed_layers={n_exec}, max_residual_diff_at_init={max_init_diff:.8f}"
    )
    assert h_up_init.shape == (1, 4, cfg.d_big)
    assert h_sub_init.shape == (1, 4, cfg.d_big)
    assert n_exec == 8
    assert max_init_diff == 0.0, f"Residual stream corrupted at init: {max_init_diff}"

    # 4. Verify Soft-Token Superposition L2 Norm Stabilization
    with torch.no_grad():
        we = engine.active_model.get_input_embeddings().weight
        sample_logits = torch.randn(1, we.shape[0], device=cfg.gpu_device, dtype=cfg.dtype) * 2.5
        e_lat, top_p, top_i = compute_soft_superposition_embedding(
            sample_logits,
            we,
            mean_embedding_l2_norm=engine.mean_embed_l2_norm,
            top_k=cfg.top_k_superposition,
            temperature=cfg.superposition_temperature,
        )
        e_norm = float(e_lat.float().norm(p=2, dim=-1).item())
        target_norm = float(engine.mean_embed_l2_norm.item())
    print(
        f"[Superposition L2 Audit] e_latent shape={tuple(e_lat.shape)} | "
        f"target ||W_E||_2={target_norm:.4f} | stabilized ||e_latent||_2={e_norm:.4f}"
    )
    assert abs(e_norm - target_norm) / max(target_norm, 1e-5) < 0.01

    # 5. Verify End-to-End Gradient Flow through Bridge & Router via SubconsciousTrainer
    trainer = SubconsciousTrainer(engine)
    train_prompt = (
        "Design a distributed lock manager across 5 consensus nodes tolerating 2 node failures. "
        "State the minimum write quorum size and fencing token invariant."
    )
    train_ids = engine.tokenizer(train_prompt, return_tensors="pt").input_ids.to(cfg.gpu_device)
    step1_metrics = trainer.train_step(train_ids, force_router_trigger=True)
    step2_metrics = trainer.train_step(train_ids, force_router_trigger=True)
    print(
        f"[Trainer Gradient Audit] Step 1 -> loss={step1_metrics.total_loss:.4f}, "
        f"bridge_grad_norm={step1_metrics.bridge_grad_norm:.4f}, "
        f"router_grad_norm={step1_metrics.router_grad_norm:.4f}, "
        f"alpha_after={step2_metrics.alpha_gate_value:.6f}"
    )
    assert step1_metrics.bridge_grad_norm > 0.0, "Zero gradient norm on LatentBridge!"
    assert step1_metrics.router_grad_norm > 0.0, "Zero gradient norm on DynamicEntropyRouter!"

    # 6. Benchmark 3 Generation Modes & Single Subconscious RAM Hop Latency
    test_query = (
        "A 5-node Raft cluster has nodes {N1, N2, N3, N4, N5} with election timeouts and log indices "
        "(term=3, idx=12) on {N1, N2, N3} and (term=2, idx=15) on {N4, N5}. If N1 crashes, can N4 win election? "
        "Give the exact formal reason."
    )

    trace_std = engine.generate_standard_cot(test_query, max_new_tokens=96)
    trace_sup = engine.generate_anchored_superposition(
        test_query, enable_subconscious_bridge=False, max_new_tokens=96
    )
    trace_sub = engine.generate_anchored_superposition(
        test_query, enable_subconscious_bridge=True, max_new_tokens=96
    )

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    print("-" * 80)
    print(
        f"[Mode A: Standard CoT]           tokens={trace_std.generated_token_count:3d} | "
        f"latency={trace_std.latency_sec:.2f}s | peak_vram={trace_std.peak_vram_gb:.3f} GB"
    )
    print(
        f"[Mode B: Caveman+Superposition]  tokens={trace_sup.generated_token_count:3d} | "
        f"latent_steps={trace_sup.latent_steps_executed} | latency={trace_sup.latency_sec:.2f}s | "
        f"peak_vram={trace_sup.peak_vram_gb:.3f} GB"
    )
    print(
        f"[Mode C: Full Subconscious RAM]  tokens={trace_sub.generated_token_count:3d} | "
        f"latent_steps={trace_sub.latent_steps_executed} | triggers={trace_sub.subconscious_triggers}/{trace_sub.total_router_evaluations} | "
        f"avg_ram_layers={trace_sub.avg_active_subconscious_layers:.2f}/8 | "
        f"latency={trace_sub.latency_sec:.2f}s | peak_vram={trace_sub.peak_vram_gb:.3f} GB"
    )
    print("-" * 80)
    print(f"[VRAM Gate Check] Overall Peak GPU VRAM = {peak_vram_gb:.3f} GB (Strict Limit: {cfg.strict_vram_limit_gb:.2f} GB)")
    assert (
        peak_vram_gb < cfg.strict_vram_limit_gb
    ), f"Peak VRAM {peak_vram_gb:.3f} GB exceeded strict limit {cfg.strict_vram_limit_gb:.2f} GB!"

    summary = {
        "status": "PASSED",
        "init_time_sec": round(init_sec, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "strict_vram_limit_gb": cfg.strict_vram_limit_gb,
        "bridge_vram_mb": round(bridge_mb, 2),
        "router_vram_mb": round(router_mb, 2),
        "subconscious_audit": sub_audit,
        "superposition_l2_norm": {"target": round(target_norm, 4), "actual": round(e_norm, 4)},
        "trainer_step1": step1_metrics.__dict__,
        "trainer_step2": step2_metrics.__dict__,
        "benchmarks": {
            "standard_cot": {
                "latency_sec": round(trace_std.latency_sec, 3),
                "tokens": trace_std.generated_token_count,
                "peak_vram_gb": round(trace_std.peak_vram_gb, 3),
                "answer_preview": trace_std.final_answer[:240],
            },
            "caveman_superposition": {
                "latency_sec": round(trace_sup.latency_sec, 3),
                "tokens": trace_sup.generated_token_count,
                "latent_steps": trace_sup.latent_steps_executed,
                "caveman_anchors": trace_sup.caveman_anchors_text,
                "peak_vram_gb": round(trace_sup.peak_vram_gb, 3),
                "answer_preview": trace_sup.final_answer[:240],
            },
            "subconscious_bridge": {
                "latency_sec": round(trace_sub.latency_sec, 3),
                "tokens": trace_sub.generated_token_count,
                "latent_steps": trace_sub.latent_steps_executed,
                "caveman_anchors": trace_sub.caveman_anchors_text,
                "triggers": trace_sub.subconscious_triggers,
                "router_evals": trace_sub.total_router_evaluations,
                "avg_active_layers": round(trace_sub.avg_active_subconscious_layers, 2),
                "peak_vram_gb": round(trace_sub.peak_vram_gb, 3),
                "answer_preview": trace_sub.final_answer[:240],
            },
        },
    }

    out_path = PROJECT_ROOT / "verify_pipeline_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Success] Full verification report saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_verification_pipeline()
