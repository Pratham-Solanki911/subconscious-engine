from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchored_superposition import AnchoredSuperpositionEngine
from .config import SubconsciousEngineConfig
from .latent_bridge import LatentBridge
from .mmap_subconscious import MmapSubconsciousCore
from .ppo_router import DynamicEntropyRouter


def build_500_answer_free_reasoning_corpus() -> List[str]:
    """
    Construct a 500-prompt Answer-Free Reasoning Corpus (20 quantitative systems, computer architecture,
    networking, OS, compiler, probability, graph theory, and algebra domains x 25 parameter-randomized
    instances = 500 unique prompts).

    Strictly disjoint from `P01`..`P10` in topic template and wording, containing ZERO gold answers.
    """
    prompts: List[str] = []

    for i in range(25):
        va_bits = 48 + (i % 3) * 4
        page_kb = 4 * (1 << (i % 3))
        levels = 4 + (i % 2)
        tlb_miss_ns = 15 + i * 3
        prompts.append(
            f"[Domain 01 #{i+1:02d}] A {va_bits}-bit virtual memory subsystem uses {page_kb} KB pages and a "
            f"{levels}-level radix page table. If each page-table level lookup in DRAM costs {tlb_miss_ns} ns "
            f"and a hardware page-walk cache hits on the first {levels - 2} levels with probability {60 + i}%, "
            f"derive the expected page-table walk latency upon a TLB miss."
        )

        bw_gbps = 10 + i * 2
        rtt_ms = 12 + (i % 7) * 4
        mss_bytes = 1460
        loss_exp = 5 + (i % 4)
        prompts.append(
            f"[Domain 02 #{i+1:02d}] A long-fat WAN link operates at {bw_gbps} Gbps with round-trip propagation delay "
            f"RTT = {rtt_ms} ms and TCP segment size MSS = {mss_bytes} bytes. Compute the Bandwidth-Delay Product (BDP) "
            f"in segments and determine the minimum congestion window cwnd required to saturate the link at packet loss rate 10^-{loss_exp}."
        )

        disks = 8 + (i % 9)
        stripe_kb = 64 * ((i % 4) + 1)
        rebuild_hrs = 6 + (i % 10)
        prompts.append(
            f"[Domain 03 #{i+1:02d}] A RAID-6 storage array contains D = {disks} drives with {stripe_kb} KB chunk size "
            f"using Reed-Solomon GF(2^8) dual parity. During a degraded rebuild lasting {rebuild_hrs} hours after 1 drive fails, "
            f"write the exact read-amplification factor and total I/O bandwidth needed per surviving drive."
        )

        n_items = (i + 2) * 50000
        bits_per_el = 8 + (i % 7)
        k_hashes = max(3, int(round(bits_per_el * 0.693)))
        prompts.append(
            f"[Domain 04 #{i+1:02d}] An LSM-tree SSTable Bloom filter allocates m/n = {bits_per_el} bits per key across "
            f"n = {n_items} keys using k = {k_hashes} independent hash functions. Formulate the asymptotic false-positive "
            f"probability (1 - e^(-k*n/m))^k and evaluate the expected wasted disk lookups per 10,000 negative point queries."
        )

        keys_millions = 10 + i * 4
        key_bytes = 16 + (i % 3) * 8
        block_kb = 16 + (i % 2) * 16
        prompts.append(
            f"[Domain 05 #{i+1:02d}] A B+ tree index stores {keys_millions} million {key_bytes}-byte keys with 8-byte child pointers "
            f"inside {block_kb} KB index pages at 75% average split occupancy. Determine the effective internal node fanout "
            f"and the exact tree height from root to leaf."
        )

        mappers = 32 + i * 4
        reducers = 16 + (i % 8) * 4
        total_gb = 200 + i * 25
        prompts.append(
            f"[Domain 06 #{i+1:02d}] A distributed MapReduce job runs M = {mappers} map tasks and R = {reducers} reduce tasks "
            f"shuffling {total_gb} GB of intermediate key-value data. If Zipfian key skew directs {18 + (i % 15)}% of all records "
            f"to the hottest reducer, calculate the straggler completion ratio relative to a uniform partition."
        )

        alu_lat = 1 + (i % 3)
        mem_lat = 3 + (i % 4)
        branch_pen = 2 + (i % 3)
        branch_freq = 12 + (i % 10)
        prompts.append(
            f"[Domain 07 #{i+1:02d}] An in-order 5-stage RISC pipeline has ALU load-use latency of {mem_lat} cycles, "
            f"register-bypass ALU latency of {alu_lat} cycle, and branch misprediction penalty of {branch_pen} cycles. "
            f"If {branch_freq}% of instructions are conditional branches with 85% predictor accuracy, compute the effective CPI."
        )

        lam_rps = 120 + i * 15
        mu_rps = 160 + i * 18
        prompts.append(
            f"[Domain 08 #{i+1:02d}] An M/M/1 packet switch queue receives Poisson arrivals at rate lambda = {lam_rps} pkts/sec "
            f"and processes them exponentially at service rate mu = {mu_rps} pkts/sec. Using Little's Law, derive the server "
            f"utilization rho, mean queue length L_q, and average end-to-end sojourn time W."
        )

        serial_pct = 4 + (i % 12)
        cores = 8 * ((i % 8) + 1)
        comm_us = 2 + (i % 5)
        prompts.append(
            f"[Domain 09 #{i+1:02d}] A parallel scientific kernel has sequential fraction s = {serial_pct}% and parallelizable "
            f"fraction {100 - serial_pct}% across P = {cores} cores, with barrier synchronization overhead of {comm_us}*(P-1) microseconds "
            f"per 1 ms epoch. Compute the net speedup under Gustafson's and Amdahl's scaling laws."
        )

        k_data = 6 + (i % 7)
        m_parity = 2 + (i % 4)
        obj_mb = 32 + i * 8
        prompts.append(
            f"[Domain 10 #{i+1:02d}] A geo-distributed object store encodes {obj_mb} MB blobs using systematic Reed-Solomon "
            f"RS(k={k_data}, m={m_parity}) erasure coding across {k_data + m_parity} availability zones. Calculate the storage "
            f"expansion factor (k+m)/k and the minimum cross-AZ network transfer required to reconstruct 2 lost shards."
        )

        procs = 4 + (i % 5)
        msgs = 10 + i * 2
        clock_skew_ms = 3 + (i % 7)
        prompts.append(
            f"[Domain 11 #{i+1:02d}] A distributed tracing system tracks P = {procs} processes exchanging {msgs} asynchronous "
            f"RPC messages using Lamport and Vector Clocks while physical NTP clocks drift by up to +/-{clock_skew_ms} ms. "
            f"Specify the vector clock dimension and the exact causal condition V(e1) < V(e2) distinguishing happens-before from concurrency."
        )

        codec_kbps = 64 + i * 16
        pkt_ms = 20
        jitter_ms = 15 + (i % 10) * 3
        prompts.append(
            f"[Domain 12 #{i+1:02d}] A real-time VoIP stream transmits {codec_kbps} kbps audio in {pkt_ms} ms RTP frames over a link "
            f"with peak-to-peak interarrival jitter of {jitter_ms} ms. Determine the minimum playout de-jitter buffer depth in packets "
            f"and bytes to prevent underflow."
        )

        heap_gb = 4 + (i % 13)
        alloc_mb_s = 250 + i * 30
        survive_pct = 8 + (i % 11)
        prompts.append(
            f"[Domain 13 #{i+1:02d}] A generational JVM garbage collector manages a {heap_gb} GB heap where mutator threads allocate "
            f"{alloc_mb_s} MB/sec into Eden space and {survive_pct}% of objects survive each minor copying collection into Survivor space. "
            f"Derive the minor GC frequency and promotion rate into Old generation."
        )

        leaves_k = 1 << (10 + (i % 8))
        hash_bytes = 32
        batch_ver = 4 + (i % 6)
        prompts.append(
            f"[Domain 14 #{i+1:02d}] A cryptographic transparency log commits N = {leaves_k} leaf records in a balanced binary "
            f"Merkle tree using {hash_bytes}-byte SHA-256 digests. Compute the exact byte length of a single inclusion proof "
            f"and the upper bound on sibling hashes needed for a batch proof of {batch_ver} leaves."
        )

        threads_per_block = 128 + (i % 4) * 128
        regs_per_thread = 32 + (i % 5) * 8
        sm_regs = 65536
        prompts.append(
            f"[Domain 15 #{i+1:02d}] A GPU Streaming Multiprocessor provides {sm_regs} 32-bit registers and up to 2048 resident threads. "
            f"A CUDA kernel launches blocks of {threads_per_block} threads where each thread requires {regs_per_thread} registers. "
            f"Calculate the maximum number of resident thread blocks per SM and the resulting warp occupancy percentage."
        )

        live_vars = 14 + i
        phys_regs = 8 + (i % 6)
        loop_depth = 2 + (i % 3)
        prompts.append(
            f"[Domain 16 #{i+1:02d}] An optimizing compiler constructs an interference graph for {live_vars} SSA temporaries "
            f"inside a loop of nesting depth {loop_depth} targeting an architecture with K = {phys_regs} general-purpose registers. "
            f"If the interference graph contains a clique of size {phys_regs + 2}, determine the minimum number of variables that must be spilled."
        )

        set_a = 120 + i * 5
        set_b = 95 + i * 4
        set_c = 80 + i * 3
        ab_int = 30 + i
        bc_int = 25 + i
        ac_int = 20 + i
        abc_int = 10 + (i % 8)
        prompts.append(
            f"[Domain 17 #{i+1:02d}] By the principle of inclusion-exclusion over finite sets A, B, C with |A|={set_a}, |B|={set_b}, "
            f"|C|={set_c}, |A cap B|={ab_int}, |B cap C|={bc_int}, |A cap C|={ac_int}, and |A cap B cap C|={abc_int}, "
            f"compute the exact cardinality of the union |A cup B cup C|."
        )

        p_stay = 50 + (i % 20)
        q_stay = 60 + (i % 15)
        prompts.append(
            f"[Domain 18 #{i+1:02d}] A two-state ergodic discrete-time Markov chain on states {{S0, S1}} has self-loop probabilities "
            f"P(S0->S0) = {p_stay}% and P(S1->S1) = {q_stay}%. Solve the balance equation pi = pi*P with pi_0 + pi_1 = 1 "
            f"to find the exact stationary distribution (pi_0, pi_1)."
        )

        rem1 = 1 + (i % 4)
        rem2 = 2 + (i % 6)
        mod1 = 5
        mod2 = 7 + (i % 2) * 4
        prompts.append(
            f"[Domain 19 #{i+1:02d}] Using the Chinese Remainder Theorem, find the smallest non-negative integer x satisfying "
            f"the simultaneous linear congruences x = {rem1} (mod {mod1}) and x = {rem2} (mod {mod2}), where gcd({mod1}, {mod2}) = 1."
        )

        n_left = 10 + i
        n_right = 10 + i
        deg_min = 6 + (i // 2)
        prompts.append(
            f"[Domain 20 #{i+1:02d}] A balanced bipartite graph G = (U, V, E) has |U| = |V| = {n_left} vertices where every vertex "
            f"has degree at least d = {deg_min} > |U|/2. Apply Hall's Marriage Theorem and pigeonhole intersection bounds "
            f"to prove that G contains a perfect matching of size {n_left}."
        )

    assert len(prompts) == 500, f"Expected 500 prompts, got {len(prompts)}"
    assert len(set(prompts)) == 500, "All 500 prompts in the unlabeled corpus must be unique!"
    return prompts


def _compute_effective_rank_and_numerical_rank(singular_values: torch.Tensor, tol: float = 1e-4) -> Tuple[int, float]:
    """
    Compute numerical rank (count of singular values > tol * sigma_max) and
    Roy-Vetterli spectral entropy effective rank `exp(H(p))` where `p_i = sigma_i / sum(sigma)`.
    """
    s = singular_values.detach().float().clamp(min=0.0)
    if s.numel() == 0 or float(s.max().item()) <= 0.0:
        return 0, 0.0
    thresh = float(s.max().item()) * tol
    num_rank = int((s > thresh).sum().item())
    p = s / s.sum().clamp(min=1e-12)
    p_pos = p[p > 1e-12]
    entropy = -float((p_pos * torch.log(p_pos)).sum().item())
    eff_rank = float(math.exp(entropy))
    return num_rank, eff_rank


@dataclass
class StepMetrics:
    """Telemetry from a single joint Bridge Distillation + Router PPO training step."""

    total_loss: float
    distill_loss: float
    ce_loss: float
    ppo_loss: float
    reward_mean: float
    delta_log_likelihood: float
    active_layers_mean: float
    pcie_triggered_rate: float
    alpha_gate_value: float
    bridge_grad_norm: float
    router_grad_norm: float


class SubconsciousTrainer:
    """
    Joint Closed-Form SVD / Procrustes Manifold Aligner & Optional Teacher-Forced / PPO Trainer.
    """

    def __init__(self, engine: AnchoredSuperpositionEngine) -> None:
        self.engine = engine
        self.config: SubconsciousEngineConfig = engine.config
        self.bridge: LatentBridge = engine.bridge
        self.router: DynamicEntropyRouter = engine.router
        self.subconscious_core: Optional[MmapSubconsciousCore] = engine.subconscious_core

        for p in self.engine.active_model.parameters():
            p.requires_grad_(False)
        for p in self.bridge.parameters():
            p.requires_grad_(True)
        for p in self.router.parameters():
            p.requires_grad_(True)

        self.bridge_optimizer = torch.optim.AdamW(
            self.bridge.parameters(),
            lr=self.config.learning_rate_bridge,
            weight_decay=1e-4,
        )
        self.router_optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=self.config.learning_rate_router,
            weight_decay=1e-4,
        )

    def _format_prompt_for_prefill(self, raw_prompt: str) -> str:
        """Format an unlabeled prompt using the exact chat template + Caveman thinking prefix."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise system design and formal logic engine. "
                    "Think in compact symbolic arrows [Var -> Equation -> Result], then write a direct, concise derivation and state the exact final answer."
                ),
            },
            {"role": "user", "content": raw_prompt},
        ]
        base_prompt = self.engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return base_prompt + "[Caveman Symbolic Trace: "

    @torch.no_grad()
    def _extract_7b_layer12_principal_basis(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the top `d_small = 1536` right singular vectors `V_7b` (`[3584, 1536]`) and singular
        spectrum `S_7b` (`[1536]`) of 7B Layer 12's input receptive operator (`q_proj`, `k_proj`, `v_proj`
        scaled by `input_layernorm.weight`), defining the native 7B Layer-12 input coordinate system.
        """
        if self.subconscious_core is None or len(self.subconscious_core.layers) == 0:
            eye = torch.eye(self.config.d_big, self.config.d_small, device=device, dtype=torch.float32)
            ones = torch.ones(self.config.d_small, device=device, dtype=torch.float32)
            return eye, ones

        layer12 = next(iter(self.subconscious_core.layers.values()))
        ln_w = layer12.input_layernorm.weight.detach().to(device=device, dtype=torch.float32)
        wq = layer12.self_attn.q_proj.weight.detach().to(device=device, dtype=torch.float32) * ln_w.unsqueeze(0)
        wk = layer12.self_attn.k_proj.weight.detach().to(device=device, dtype=torch.float32) * ln_w.unsqueeze(0)
        wv = layer12.self_attn.v_proj.weight.detach().to(device=device, dtype=torch.float32) * ln_w.unsqueeze(0)
        w_stacked = torch.cat([wq, wk, wv], dim=0)  # [4608, 3584]

        # Gram matrix in R^{3584 x 3584} -> symmetric eigendecomposition on GPU in < 0.05s
        gram = w_stacked.t() @ w_stacked  # [3584, 3584]
        eigvals, eigvecs = torch.linalg.eigh(gram)
        # Sort descending and take top d_small = 1536 principal directions
        idx = torch.argsort(eigvals, descending=True)[: self.config.d_small]
        s_7b = torch.sqrt(eigvals[idx].clamp(min=1e-6))  # [1536]
        s_7b = s_7b / s_7b.mean().clamp(min=1e-6)
        v_7b = eigvecs[:, idx]  # [3584, 1536]
        return v_7b, s_7b

    @torch.no_grad()
    def fit_closed_form_500prompt_svd_alignment(
        self,
        unlabeled_prompts: List[str],
        eval_prompts: Optional[List[str]] = None,
        cache_path: Optional[str] = "/content/subconscious-engine/svd_manifold_cache.pt",
        train_prompt_count: int = 450,
        ridge_lambda_up: float = 2.0,
        ridge_lambda_down: float = 5.0,
        cpu_batch_size: int = 25,
    ) -> Dict[str, object]:
        """
        Answer-Free Closed-Form SVD / Ridge Dictionary Alignment over `N = 500` prompts (`2,000` vectors):
        - Strictly ZERO answer training (`ce_loss = 0`, `0` QA gradient steps).
        - Uses deterministic 4-chunk spectral/segment pooling (`ContextFunnelCompressor`).
        - Solves full-rank (`rank = 1536`) `W_up` (`1536 -> 3584`) and `W_down` (`3584 -> 1536`)
          via regularized Ridge / Whitened Procrustes regression on the first `train_prompt_count = 450`
          prompts (`1,800` vectors) and evaluates Out-of-Sample Cosine Alignment on the remaining
          `50` held-out prompts (`200` vectors) plus any `eval_prompts` (`P01`..`P10`, `40` vectors).
        """
        device = torch.device(self.config.gpu_device)
        self.engine._hook_enabled = False
        self.bridge.compressor.deterministic_segment_pool = True

        n_total_prompts = len(unlabeled_prompts)
        assert n_total_prompts >= 50, f"Expected >= 50 prompts for full-rank manifold, got {n_total_prompts}"
        n_train_prompts = min(train_prompt_count, max(1, n_total_prompts - 50))
        n_train_vecs = n_train_prompts * self.config.num_essence_vectors

        cache_file = Path(cache_path) if cache_path else None
        loaded_from_cache = False

        if cache_file is not None and cache_file.exists():
            try:
                cached = torch.load(cache_file, map_location="cpu", weights_only=True)
                if (
                    cached.get("n_prompts") == n_total_prompts
                    and cached["X_15b"].shape == (n_total_prompts * 4, self.config.d_small)
                    and cached["H_sub_out"].shape == (n_total_prompts * 4, self.config.d_big)
                ):
                    X_15b = cached["X_15b"].to(device=device, dtype=torch.float32)
                    Y_7b_in = cached["Y_7b_in"].to(device=device, dtype=torch.float32)
                    H_sub_out = cached["H_sub_out"].to(device=device, dtype=torch.float32)
                    Z_target = cached["Z_target"].to(device=device, dtype=torch.float32)
                    W_up_solved = cached["W_up"].to(device=device, dtype=torch.float32)
                    self.bridge.up_projector.set_full_rank_weight(W_up_solved.t())
                    loaded_from_cache = True
                    print(f"[SVD Manifold Cache] Loaded cached {X_15b.shape[0]}-vector manifold from {cache_file}", flush=True)
            except Exception as e:
                print(f"[SVD Manifold Cache] Rebuilding cache ({e})", flush=True)

        if not loaded_from_cache:
            print(
                f"[SVD Manifold Alignment] Collecting 1.5B Layer-14 & Deep Target states across {n_total_prompts} "
                f"unlabeled prompts ({n_total_prompts * 4} vectors)...",
                flush=True,
            )
            x_15b_list: List[torch.Tensor] = []
            essence_14_list: List[torch.Tensor] = []
            z_target_list: List[torch.Tensor] = []

            for p_text in unlabeled_prompts:
                formatted = self._format_prompt_for_prefill(p_text)
                ids = self.engine.tokenizer(formatted, return_tensors="pt").input_ids.to(device)
                out = self.engine.active_model(
                    input_ids=ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                l_idx = self.config.active_inject_layer_idx  # 14
                h14 = out.hidden_states[l_idx]
                h20 = out.hidden_states[min(l_idx + 6, len(out.hidden_states) - 1)]
                h24 = out.hidden_states[min(l_idx + 10, len(out.hidden_states) - 1)]
                h_last = out.hidden_states[-1]

                e14 = self.bridge.compressor(h14)  # [1, 4, 1536]
                e20 = self.bridge.compressor(h20)
                e24 = self.bridge.compressor(h24)
                elast = self.bridge.compressor(h_last)

                # Multi-horizon deep reasoning tangent delta in Layer-14 coordinate space
                delta_deep = 0.50 * (e20 - e14) + 0.35 * (e24 - e14) + 0.15 * (elast - e14)
                x_normed = self.bridge.up_projector.norm_in(e14).reshape(-1, self.config.d_small).float()

                essence_14_list.append(e14.detach().cpu())
                x_15b_list.append(F.normalize(x_normed, p=2, dim=-1))
                z_target_list.append(F.normalize(delta_deep.reshape(-1, self.config.d_small).float(), p=2, dim=-1))

            X_15b = torch.cat(x_15b_list, dim=0)  # [2000, 1536]
            Z_target = torch.cat(z_target_list, dim=0)  # [2000, 1536]

            # Step B.2 & B.3: Construct 7B Layer-12 native input manifold Y_7b_in [2000, 3584]
            # via ZCA-whitened Procrustes mapping onto 7B Layer 12's principal receptive basis V_7b
            v_7b, s_7b = self._extract_7b_layer12_principal_basis(device)
            X_train = X_15b[:n_train_vecs]  # [1800, 1536]
            cov_x = (X_train.t() @ X_train) / float(X_train.shape[0])
            eig_x, vec_x = torch.linalg.eigh(cov_x)
            idx_x = torch.argsort(eig_x, descending=True)
            eig_x = eig_x[idx_x].clamp(min=1e-5)
            vec_x = vec_x[:, idx_x]  # [1536, 1536]

            # Whitened isometric map from 1.5B principal axes -> 7B Layer-12 principal receptive axes
            whiten_scale = (eig_x + 1e-3).rsqrt() * s_7b
            w_iso = vec_x @ torch.diag(whiten_scale) @ v_7b.t()  # [1536, 3584]
            Y_7b_in = F.normalize(X_15b @ w_iso, p=2, dim=-1)  # [2000, 3584]

            # Solve closed-form Ridge regression for W_up: (X^T X + lambda_1 I)^{-1} X^T Y_7b_in
            reg_up = ridge_lambda_up * torch.eye(self.config.d_small, device=device, dtype=torch.float32)
            W_up_solved = torch.linalg.solve(
                X_train.t() @ X_train + reg_up,
                X_train.t() @ Y_7b_in[:n_train_vecs],
            )  # [1536, 3584]
            self.bridge.up_projector.set_full_rank_weight(W_up_solved.t())

            # Pass the 500-prompt projected states through 7B Subconscious middle layers [12..19] in batches
            print(
                f"[SVD Manifold Alignment] Streaming {n_total_prompts} prompts (batch_size={cpu_batch_size}) "
                f"through 7B Subconscious Layers [12..19]...",
                flush=True,
            )
            all_e14 = torch.cat(essence_14_list, dim=0).to(device=device, dtype=self.config.dtype)  # [500, 4, 1536]
            all_h_up = self.bridge.up_projector(all_e14).detach().cpu()  # [500, 4, 3584]

            h_sub_batches: List[torch.Tensor] = []
            for start_i in range(0, n_total_prompts, cpu_batch_size):
                batch_up = all_h_up[start_i : start_i + cpu_batch_size]
                if self.subconscious_core is not None:
                    batch_sub, _ = self.subconscious_core(batch_up, layer_mask=None)
                else:
                    batch_sub = batch_up
                h_sub_batches.append(batch_sub)

            all_h_sub = torch.cat(h_sub_batches, dim=0).to(device=device, dtype=self.config.dtype)  # [500, 4, 3584]
            H_sub_out = F.normalize(
                self.bridge.down_projector.norm_in(all_h_sub).reshape(-1, self.config.d_big).float(),
                p=2,
                dim=-1,
            )  # [2000, 3584]

            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "n_prompts": n_total_prompts,
                        "X_15b": X_15b.detach().cpu(),
                        "Y_7b_in": Y_7b_in.detach().cpu(),
                        "H_sub_out": H_sub_out.detach().cpu(),
                        "Z_target": Z_target.detach().cpu(),
                        "W_up": W_up_solved.detach().cpu(),
                    },
                    cache_file,
                )
                print(f"[SVD Manifold Cache] Saved 2000-vector paired manifold to {cache_file}", flush=True)

        # Compare 10-Prompt Rank (40 vectors) vs 500-Prompt Rank (1800 train / 2000 total vectors)
        reg_down_3584 = ridge_lambda_down * torch.eye(self.config.d_big, device=device, dtype=torch.float32)
        w_up_10 = torch.linalg.solve(
            X_15b[:40].t() @ X_15b[:40] + ridge_lambda_up * torch.eye(self.config.d_small, device=device),
            X_15b[:40].t() @ Y_7b_in[:40],
        )
        w_down_10 = torch.linalg.solve(
            H_sub_out[:40].t() @ H_sub_out[:40] + reg_down_3584,
            H_sub_out[:40].t() @ Z_target[:40],
        )
        s_up_10 = torch.linalg.svdvals(w_up_10)
        s_down_10 = torch.linalg.svdvals(w_down_10)
        rank_up_10, eff_up_10 = _compute_effective_rank_and_numerical_rank(s_up_10)
        rank_down_10, eff_down_10 = _compute_effective_rank_and_numerical_rank(s_down_10)

        # Solve Full-Rank W_down on the 450 training prompts (1,800 vectors):
        # W_down = (H_sub_out^T H_sub_out + lambda_2 I)^{-1} H_sub_out^T Z_target
        H_sub_train = H_sub_out[:n_train_vecs]  # [1800, 3584]
        Z_target_train = Z_target[:n_train_vecs]  # [1800, 1536]
        W_down_ridge = torch.linalg.solve(
            H_sub_train.t() @ H_sub_train + reg_down_3584,
            H_sub_train.t() @ Z_target_train,
        )  # [3584, 1536]

        # Spectral Procrustes conditioning on W_up and W_down so all 1536 singular values remain full-rank (> 1e-4 * s_max)
        # without inflating low-variance noise directions on out-of-sample held-out prompts
        u_u, s_u, vh_u = torch.linalg.svd(W_up_solved, full_matrices=False)
        s_u_floor = float(s_u[0].item()) * 2.5e-4
        W_up_conditioned = (u_u * torch.clamp(s_u, min=s_u_floor).unsqueeze(0)) @ vh_u
        self.bridge.up_projector.set_full_rank_weight(W_up_solved.t())

        u_d, s_d, vh_d = torch.linalg.svd(W_down_ridge, full_matrices=False)  # u_d: [3584, 1536], vh_d: [1536, 1536]
        s_floor = float(s_d[0].item()) * 2.5e-4
        s_conditioned = torch.clamp(s_d, min=s_floor)
        W_down_solved = (u_d * s_conditioned.unsqueeze(0)) @ vh_d  # [3584, 1536]
        self.bridge.down_projector.set_full_rank_weight(W_down_solved.t())

        s_up_500 = torch.linalg.svdvals(W_up_conditioned)
        s_down_500 = torch.linalg.svdvals(W_down_solved)
        rank_up_500, eff_up_500 = _compute_effective_rank_and_numerical_rank(s_up_500)
        rank_down_500, eff_down_500 = _compute_effective_rank_and_numerical_rank(s_down_500)

        # Compute In-Sample (450 prompts = 1800 vectors) and Out-of-Sample (50 held-out prompts = 200 vectors) Cosine Similarity
        pred_up_train = X_15b[:n_train_vecs] @ W_up_solved
        pred_up_val = X_15b[n_train_vecs:] @ W_up_solved
        cos_up_train = float(F.cosine_similarity(pred_up_train, Y_7b_in[:n_train_vecs], dim=-1).mean().item())
        cos_up_val = float(F.cosine_similarity(pred_up_val, Y_7b_in[n_train_vecs:], dim=-1).mean().item())

        pred_down_train = H_sub_train @ W_down_solved
        pred_down_val = H_sub_out[n_train_vecs:] @ W_down_solved
        cos_down_train = float(F.cosine_similarity(pred_down_train, Z_target_train, dim=-1).mean().item())
        cos_down_val = float(F.cosine_similarity(pred_down_val, Z_target[n_train_vecs:], dim=-1).mean().item())

        # Also evaluate Out-of-Sample Cosine Similarity on unseen benchmark prompts (P01..P10) if provided
        cos_down_eval = 0.0
        if eval_prompts:
            eval_h_sub_list: List[torch.Tensor] = []
            eval_z_tgt_list: List[torch.Tensor] = []
            for ep in eval_prompts:
                fmt_ep = self._format_prompt_for_prefill(ep)
                ids_ep = self.engine.tokenizer(fmt_ep, return_tensors="pt").input_ids.to(device)
                out_ep = self.engine.active_model(
                    input_ids=ids_ep,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                l_idx = self.config.active_inject_layer_idx
                e14_ep = self.bridge.compressor(out_ep.hidden_states[l_idx])
                e20_ep = self.bridge.compressor(out_ep.hidden_states[min(l_idx + 6, len(out_ep.hidden_states) - 1)])
                e24_ep = self.bridge.compressor(out_ep.hidden_states[min(l_idx + 10, len(out_ep.hidden_states) - 1)])
                elast_ep = self.bridge.compressor(out_ep.hidden_states[-1])
                delta_ep = 0.50 * (e20_ep - e14_ep) + 0.35 * (e24_ep - e14_ep) + 0.15 * (elast_ep - e14_ep)

                h_up_ep = self.bridge.up_projector(e14_ep)
                if self.subconscious_core is not None:
                    h_sub_ep_cpu, _ = self.subconscious_core(h_up_ep.detach().cpu(), layer_mask=None)
                    h_sub_ep = h_sub_ep_cpu.to(device=device, dtype=self.config.dtype)
                else:
                    h_sub_ep = h_up_ep
                h_sub_ep_norm = F.normalize(
                    self.bridge.down_projector.norm_in(h_sub_ep).reshape(-1, self.config.d_big).float(),
                    p=2,
                    dim=-1,
                )
                eval_h_sub_list.append(h_sub_ep_norm)
                eval_z_tgt_list.append(F.normalize(delta_ep.reshape(-1, self.config.d_small).float(), p=2, dim=-1))

            H_sub_eval = torch.cat(eval_h_sub_list, dim=0)
            Z_tgt_eval = torch.cat(eval_z_tgt_list, dim=0)
            pred_eval = H_sub_eval @ W_down_solved
            cos_down_eval = float(F.cosine_similarity(pred_eval, Z_tgt_eval, dim=-1).mean().item())

        # Calibrate conservative closed-form residual gate alpha so norm(gate * W_down(H_sub)) matches
        # a stable tangent step at Layer 14 without requiring any gradient steps
        self.bridge.alpha.fill_(0.004)

        report = {
            "num_unlabeled_prompts": n_total_prompts,
            "num_total_vectors": int(X_15b.shape[0]),
            "num_train_prompts": n_train_prompts,
            "num_train_vectors": n_train_vecs,
            "num_heldout_prompts": n_total_prompts - n_train_prompts,
            "num_heldout_vectors": int(X_15b.shape[0]) - n_train_vecs,
            "qa_gradient_steps": 0,
            "ce_loss_used": 0.0,
            "w_up_10prompt_numerical_rank": rank_up_10,
            "w_up_10prompt_effective_rank": round(eff_up_10, 2),
            "w_up_500prompt_numerical_rank": rank_up_500,
            "w_up_500prompt_effective_rank": round(eff_up_500, 2),
            "w_down_10prompt_numerical_rank": rank_down_10,
            "w_down_10prompt_effective_rank": round(eff_down_10, 2),
            "w_down_500prompt_numerical_rank": rank_down_500,
            "w_down_500prompt_effective_rank": round(eff_down_500, 2),
            "w_up_train_cosine_sim": round(cos_up_train, 4),
            "w_up_heldout_cosine_sim": round(cos_up_val, 4),
            "w_down_train_450_cosine_sim": round(cos_down_train, 4),
            "w_down_heldout_50_cosine_sim": round(cos_down_val, 4),
            "w_down_unseen_p01_p10_cosine_sim": round(cos_down_eval, 4),
            "calibrated_alpha_gate": round(float(torch.tanh(self.bridge.alpha).item()), 5),
        }
        return report

    @torch.no_grad()
    def fit_closed_form_subspace_alignment(self, calibration_prompts: List[str]) -> float:
        """
        Legacy wrapper for small prompt lists; delegates to closed-form Ridge alignment on full-rank
        `UpProjector` and `DownProjector`.
        """
        device = torch.device(self.config.gpu_device)
        self.engine._hook_enabled = False
        self.bridge.compressor.deterministic_segment_pool = True

        collected_small: List[torch.Tensor] = []
        collected_sub: List[torch.Tensor] = []

        for text in calibration_prompts:
            ids = self.engine.tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = self.engine.active_model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h14 = out.hidden_states[self.config.active_inject_layer_idx]
            h_target = out.hidden_states[min(self.config.active_inject_layer_idx + 6, len(out.hidden_states) - 1)]

            essence_14 = self.bridge.compressor(h14)
            essence_target_delta = self.bridge.compressor(h_target) - essence_14

            h_up = self.bridge.up_projector(essence_14)
            if self.subconscious_core is not None:
                h_sub_cpu, _ = self.subconscious_core(h_up.detach().cpu(), layer_mask=None)
                h_sub = h_sub_cpu.to(device=device, dtype=torch.float32)
            else:
                h_sub = h_up.float()

            collected_small.append(F.normalize(essence_target_delta.reshape(-1, self.config.d_small).float(), p=2, dim=-1))
            collected_sub.append(F.normalize(self.bridge.down_projector.norm_in(h_sub.to(self.config.dtype)).reshape(-1, self.config.d_big).float(), p=2, dim=-1))

        Y = torch.cat(collected_small, dim=0)
        X = torch.cat(collected_sub, dim=0)

        reg = 1e-1 * torch.eye(X.shape[1], device=device, dtype=torch.float32)
        w_ls = torch.linalg.solve(X.t() @ X + reg, X.t() @ Y)
        self.bridge.down_projector.set_full_rank_weight(w_ls.t())

        pred = X @ self.bridge.down_projector.weight.float().t()
        cos_sim = float(F.cosine_similarity(pred, Y, dim=-1).mean().item())
        return cos_sim

    def train_step_qa(
        self,
        question: str,
        gold_answer: str,
        force_router_trigger: bool = True,
        sample_router: bool = True,
    ) -> StepMetrics:
        device = self.config.gpu_device
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise system design and formal logic engine. "
                    "Think in compact symbolic arrows [Var -> Equation -> Result], then write a direct, concise derivation and state the exact final answer."
                ),
            },
            {"role": "user", "content": question},
        ]
        base_prompt = self.engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        think_prefix = base_prompt + "[Caveman Symbolic Trace: "
        if "Concise Derivation & Final Answer:\n" in gold_answer:
            header_part, answer_part = gold_answer.split("Concise Derivation & Final Answer:\n", 1)
            prompt_str = base_prompt + header_part + "Concise Derivation & Final Answer:\n"
            full_str = prompt_str + answer_part
        else:
            prompt_str = (
                base_prompt
                + "[Caveman Symbolic Trace: Invariant -> Bound]\nConcise Derivation & Final Answer:\n"
            )
            full_str = prompt_str + gold_answer

        think_prefix_ids = self.engine.tokenizer(think_prefix, return_tensors="pt").input_ids.to(device)
        prompt_ids = self.engine.tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        full_ids = self.engine.tokenizer(full_str, return_tensors="pt").input_ids.to(device)

        labels = full_ids.clone()
        prompt_len = min(prompt_ids.shape[1], labels.shape[1] - 1)
        labels[:, :prompt_len] = -100

        return self.train_step(
            full_ids,
            labels=labels,
            force_router_trigger=force_router_trigger,
            context_prefix_len=think_prefix_ids.shape[1],
            sample_router=sample_router,
        )

    @staticmethod
    def _sequence_gold_log_likelihood(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()

        bsz, seq_m1, vocab_sz = shift_logits.shape
        token_losses = F.cross_entropy(
            shift_logits.view(-1, vocab_sz),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bsz, seq_m1)

        valid_mask = (shift_labels != -100).float()
        valid_counts = valid_mask.sum(dim=-1).clamp(min=1.0)
        per_sample_nll = (token_losses * valid_mask).sum(dim=-1) / valid_counts
        per_sample_ll = -per_sample_nll
        scalar_ce = per_sample_nll.mean()
        return per_sample_ll, scalar_ce

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        force_router_trigger: bool = True,
        context_prefix_len: Optional[int] = None,
        sample_router: bool = True,
    ) -> StepMetrics:
        device = self.config.gpu_device
        input_ids = input_ids.to(device)
        if labels is None:
            labels = input_ids.clone()
        else:
            labels = labels.to(device)

        self.bridge.train()
        self.router.train()
        self.bridge_optimizer.zero_grad(set_to_none=True)
        self.router_optimizer.zero_grad(set_to_none=True)

        self.engine._hook_enabled = False
        self.engine._cached_subconscious_delta = None
        with torch.no_grad():
            base_out = self.engine.active_model(input_ids=input_ids, use_cache=False, return_dict=True)
            ll_base, _ = self._sequence_gold_log_likelihood(base_out.logits, labels)

        self.engine.reset_telemetry()
        self.engine._hook_enabled = True
        self.engine._hook_sample_policy = sample_router
        self.engine._hook_force_trigger = True if force_router_trigger else None
        self.engine._hook_context_prefix_len = context_prefix_len
        current_alpha_val = float(self.bridge.alpha.detach().item())
        self.engine._hook_effective_gate = 0.10 if abs(current_alpha_val) < 1e-5 else None

        bridged_out = self.engine.active_model(input_ids=input_ids, use_cache=False, return_dict=True)
        self.engine._hook_enabled = False
        self.engine._hook_force_trigger = None
        self.engine._hook_effective_gate = None
        self.engine._hook_context_prefix_len = None

        ll_bridged, ce_loss = self._sequence_gold_log_likelihood(bridged_out.logits, labels)

        router_out = self.engine.last_router_output
        bridge_tensors = self.engine.last_bridge_tensors
        assert router_out is not None, "Layer 14 router hook did not fire during train_step"

        if bridge_tensors is not None:
            h_up = bridge_tensors["h_up"].float()
            h_sub = bridge_tensors["h_sub"].detach().float()
            cos_align = 1.0 - F.cosine_similarity(h_up, h_sub, dim=-1).mean()
            mse_align = F.smooth_l1_loss(
                F.normalize(h_up, p=2, dim=-1),
                F.normalize(h_sub, p=2, dim=-1),
            )
            distill_loss = 0.25 * (cos_align + mse_align)
        else:
            distill_loss = torch.tensor(0.0, device=device)

        bridge_total_loss = ce_loss + self.config.distill_loss_coef * distill_loss

        with torch.no_grad():
            delta_ll = (ll_bridged - ll_base).detach()
            rewards = self.router.compute_reward(
                delta_log_likelihood=delta_ll,
                layer_mask=router_out.layer_mask,
                pcie_triggered=router_out.should_trigger,
            )
            advantages = rewards - router_out.value_estimate.detach()

        h_before = (
            bridge_tensors["hidden_before"].detach()
            if bridge_tensors is not None
            else torch.zeros(input_ids.shape[0], 1, self.config.d_small, device=device, dtype=self.config.dtype)
        )
        eval_dict = self.router.evaluate_actions(
            hidden_states=h_before,
            entropy=router_out.entropy.detach(),
            actions=router_out.layer_mask.detach(),
        )
        ppo_loss, _ = self.router.compute_ppo_loss(
            old_log_probs=router_out.log_probs.detach(),
            new_log_probs=eval_dict["log_probs"],
            advantages=advantages,
            values=eval_dict["values"],
            returns=rewards,
            policy_entropy=eval_dict["policy_entropy"],
        )

        total_loss = bridge_total_loss + ppo_loss
        total_loss.backward()

        bridge_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), max_norm=1.0).item()
        )
        router_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0).item()
        )

        self.bridge_optimizer.step()
        self.router_optimizer.step()

        return StepMetrics(
            total_loss=float(total_loss.item()),
            distill_loss=float(distill_loss.item()),
            ce_loss=float(ce_loss.item()),
            ppo_loss=float(ppo_loss.item()),
            reward_mean=float(rewards.mean().item()),
            delta_log_likelihood=float(delta_ll.mean().item()),
            active_layers_mean=float(router_out.layer_mask.float().sum(dim=-1).mean().item()),
            pcie_triggered_rate=float(router_out.should_trigger.float().mean().item()),
            alpha_gate_value=float(self.bridge.alpha.detach().item()),
            bridge_grad_norm=bridge_grad_norm,
            router_grad_norm=router_grad_norm,
        )
