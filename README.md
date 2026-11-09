# Subconscious Latent Offloading (`subconscious-engine`)

**Single-Shot System-RAM Steering Across Quantized and Cross-Scale Language Models**

**Author:** Pratham Solanki (`prathamsolanki.p28@gmail.com`)

---

## Overview

Standard layer-offloading inference engines (`llama.cpp`, `accelerate`, `FlexGen`) execute offloaded CPU/RAM transformer blocks sequentially on **every generated token**, stalling autoregressive decoding on PCIe/DDR memory bandwidth (`2–5 tokens/s`).

**Subconscious Latent Offloading** decouples a fast, self-contained GPU model (**the Active Mouth**, strictly `< 3.2 GB` VRAM) from a memory-mapped System-RAM slice of middle reasoning layers $\mathcal{L}_{12..19}$ (**the Subconscious Brain**), invoking the System-RAM slice **strictly once per query** on $k = 4$ compressed latent vectors (`12 KB`) during prompt prefill. By overlapping this single-shot CPU pass asynchronously behind GPU prefill tail layers (`L14..27`) and hybrid symbolic-continuous reasoning cycles (*Caveman Symbolic Anchors + Soft-Token Superposition*), effective GPU blocking wait time drops to **8–12 microseconds (`0.008–0.012 ms`)**.

---

## Core Contributions & Operating Regimes

1. **Deterministic 4-Segment Context Funnel (`src/latent_bridge.py`):**
   Compresses arbitrary-length Layer-14 prompt activations $H^{(14)} \in \mathbb{R}^{S \times d_{\text{gpu}}}$ into $k = 4$ vectors (`12 KB` in FP16: three quartile mean states plus the terminal instruction token state) with zero trainable pooling parameters.
2. **Intra-Layer Outlier-Aware Mixed Precision (`1% FP16 + 99% INT4`, `src/mmap_subconscious.py`):**
   `IntraLayerSplitQwen2MLP` isolates the top `1%` outlier weight columns in contiguous FP16 and packs the remaining `99%` bulk weights into per-channel INT4 (`uint8` packed) with pre-quantization activation outlier masking (`index_fill_`). This compresses the 7B System-RAM slice by **3.48× (`2,816 MB -> 808.15 MB`)**, accelerates CPU execution by **2.34×–2.69× (`125.0 ms -> 45.9 ms`)**, and preserves **0.9906–0.9993** cosine similarity against full 16-bit precision.
3. **Methodology I — Zero-Calibration Same-Model Precision Bridge ($W_{\text{up}} = I_{1536}, W_{\text{down}} = I_{1536}$):**
   When an edge GPU runs middle layers $\mathcal{L}_{12..19}$ at aggressive 3-bit per-tensor symmetric quantization (`2.52–2.57 GB` effective VRAM) while System RAM holds the crisp FP16 middle layers of the *exact same checkpoint*, their latent basis vectors are unrotated ($W_{\text{up}} = I, W_{\text{down}} = I$). With **0 calibration prompts, 0 SVD solves, and 0 trainable bridge parameters (`0.00 MB`)**, a single asynchronous System-RAM pass restores zero-shot accuracy from **10.0% (`0.0%` unbridged Caveman) to 60.0%** and reduces gold-solution perplexity from **9.045 to 5.754** (`+0.4523 nats/token`).
4. **Methodology II — Cross-Scale Subspace Translator (`1.5B GPU <-> 7B Split RAM`):**
   Bridges distinct model scales ($d_{\text{gpu}} = 1536 \leftrightarrow d_{\text{ram}} = 3584$) via an **Answer-Free Closed-Form SVD / Ridge-Procrustes Alignment** over 500 unlabeled prompts (`2,000` vectors, `0` QA gradient steps, $\mathcal{L}_{\text{CE}} = 0.0000$), achieving full numerical rank (`1536 / 1536`) and **0.9903** held-out cosine similarity with zero overfitting gap, and reaching **100% (`10/10`) accuracy in `2.88s`** when lightly task-calibrated.

---

## Experimental Results Summary

### Table 1: Methodology I — Zero-Calibration Same-Model Precision Bridge ($W = I$, `0` Calibration Prompts)

| Mode / Configuration | Calib. Prompts | Bridge Params | $W_{\text{up}}, W_{\text{down}}$ | Gold LL (`nats/tok`) | Perplexity (`PPL`) | Zero-Shot Accuracy (`P01`–`P10`) | Avg Tokens | Avg Latency (`s`) | Effective GPU Wait (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(1a) Low-Bit Quantized GPU (Standard CoT)** | `0` | `0 MB` | None | `-2.2023` | `9.045` | `10.0% (1/10)` | `160.0` | `5.31s` | `0.000 ms` |
| **(1b) Low-Bit Quantized GPU (Caveman Unbridged)** | `0` | `0 MB` | None | `-2.2023` | `9.045` | `0.0% (0/10)` | `228.0` | `7.80s` | `0.000 ms` |
| **(1c) Low-Bit GPU + Crisp FP16 RAM Bridge ($W = I$)** | **`0`** | **`0 MB`** | **Identity ($W = I_{1536}$)** | **`-1.7500` (`+0.4523`)** | **`5.754`** | **`60.0% (6/10)`** | **`114.8`** | **`4.13s`** | **`0.012 ms`** |
| **(1d) Unquantized Crisp FP16 GPU Upper Bound** | `0` | `0 MB` | None | `-1.1585` | `3.185` | `70.0% (7/10)` | `111.1` | `3.94s` | `0.000 ms` |

### Table 2: Methodology II — Cross-Scale (`1.5B GPU + 7B Split System RAM`) Reasoning Benchmark (`P01`–`P10`)

| Cross-Scale Mode (`1.5B GPU + 7B Split RAM`) | QA Steps | 7B RAM Size | Active RAM Layers | GPU Wait (`ms`) | Avg Tokens | Avg Latency (`s`) | Solved (`P01`–`P10`) | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(2a) 1.5B Standard CoT (GPU Only)** | `0` | `0 MB` | `0.0 / 8` | `0.000 ms` | `106.3` | `3.55s` | `5/10` | `50.0%` |
| **(2b) 1.5B Caveman + Superposition (GPU Only)** | `0` | `0 MB` | `0.0 / 8` | `0.000 ms` | `111.1` | `3.94s` | `7/10` | `70.0%` |
| **(2c-i) Pure 500p SVD Bridge ($H_{\text{sub\_out}}$ Target, `ablation_run15`)** | `0` | `808 MB` | `7.9 / 8` | `0.008 ms` | `101.5` | `3.67s` | `6/10` | `60.0%` (`7/10` union) |
| **(2c-ii) Pure 500p SVD Bridge ($Y_{\text{disp}}$ Target, `ablation_run14`)** | `0` | `808 MB` | `7.2 / 8` | `0.010 ms` | `116.9` | `4.38s` | `6/10` | `60.0%` (`7/10` union) |
| **(2d) Disjoint External OOD Bridge (`ablation_run13`)** | `Ext.` | `808 MB` | `5.0 / 8` | `0.009 ms` | `96.4` | `3.12s` | `6/10` | `60.0%` |
| **(2e) Task-Calibrated Subconscious Bridge (`ablation_run12/13`)** | `Calib.` | `808 MB` | `5.0 / 8` | **`0.009 ms`** | **`74.1`** | **`2.88s`** | **`10/10`** | **`100.0%`** |

---

## Repository Structure

```text
subconscious-engine/
├── src/
│   ├── config.py                  # Hardware & model hyperparameters (< 6GB GPU VRAM budget)
│   ├── latent_bridge.py           # 4-Segment Funnel, Full-Rank SVD Projectors, & SameModelPrecisionBridge (W = I)
│   ├── mmap_subconscious.py       # Memory-mapped 7B middle layers [12..19] & IntraLayerSplitQwen2MLP (1% FP16 + 99% INT4)
│   ├── anchored_superposition.py  # Caveman Symbolic Anchors, Soft-Token Superposition, & Async ThreadPool overlap
│   ├── ppo_router.py              # Dynamic Entropy Router & representation velocity early exit
│   └── trainer.py                 # 500-prompt answer-free closed-form SVD/Ridge solver & optional bridge calibration
├── benchmarks/
│   ├── run_ablation.py            # End-to-end P01..P10 evaluation harness for Methodology 1 & Methodology 2
│   └── results/
│       └── ablation_results.json  # Per-problem generation traces and hardware telemetry
├── tests/
│   ├── test_vram_budget.py        # GPU VRAM ceiling tripwire (< 5.20 GB)
│   └── verify_pipeline.py         # End-to-end shape, zero-corruption init, and router tripwires
├── paper1_results.json            # Unified Paper #1 experimental tables and microbenchmark metrics
└── ablation_run*.log              # Full execution logs from L4 GPU + System RAM benchmark runs
```

---

## Quickstart & Reproduction

```bash
# 1. Install dependencies
pip install -e .

# 2. Run local architecture & zero-corruption initialization tripwires
python -m src.latent_bridge
python -m tests.verify_pipeline

# 3. Run the complete Paper #1 benchmark suite (Methodology 1 & Methodology 2)
python -m benchmarks.run_ablation
```
