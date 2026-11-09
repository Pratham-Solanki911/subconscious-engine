# 🧠 Subconscious Latent Offloading

**Single-Shot System-RAM Steering Across Quantized and Cross-Scale Language Models**

[![Preprint](https://img.shields.io/badge/Preprint-2026-b31b1b.svg)](#-citation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> *"Your brain does the thinking; your mouth speaks it. Nobody rewires the core reasoning brain just to speak quickly, nor does the mouth need a 7B parameter pass to choose between a comma and a period."*

---

## ⚡ Overview

Standard LLM layer-offloading engines (`llama.cpp`, `accelerate`, `FlexGen`) execute offloaded CPU/System-RAM layers on **every generated token**, choking autoregressive decoding down to `2–5 tokens/s` over PCIe/DDR bandwidth.

**Subconscious Latent Offloading** decouples inference into two asynchronous streams:

1. **The Active GPU Mouth (`Qwen2.5-1.5B` or Low-Bit Quantized Checkpoint):** Runs 100% autonomously inside GPU VRAM (`2.57–3.13 GB`) during autoregressive token generation.
2. **The Subconscious System-RAM Brain (`Layers 12..19`):** Invoked **strictly once per query** at Layer 14 of prompt prefill on `k = 4` funnel-compressed latent vectors (`12 KB`).

By overlapping this single-shot CPU pass asynchronously behind GPU prefill tail layers (`L14..27`) and two hybrid **Caveman + Soft-Token Superposition** cycles, effective GPU blocking wait time drops to **8–12 microseconds (`0.008–0.012 ms`)**.

---

## 🏆 Key Results & Hardware Benchmarks

### 1. Methodology I: Zero-Calibration Same-Model Precision Bridge ($\mathbf{W}_{\text{up}} = \mathbf{I}, \mathbf{W}_{\text{down}} = \mathbf{I}$)

When an edge GPU runs an aggressively low-bit quantized model (3-bit symmetric middle layers `L12..19`) while System RAM holds the crisp FP16 middle layers of the **exact same checkpoint**, their coordinate bases are identical ($\mathbf{W} = \mathbf{I}_{1536}$). Requiring **0 calibration prompts, 0 SVD solves, and 0 trained weights (`0 MB`)**, a single asynchronous RAM pass heals quantization collapse:

| Mode / Configuration | Calib. Prompts | Trained Params | Gold LL (nats/tok) | Perplexity (PPL) ↓ | Zero-Shot Accuracy (`P01–P10`) | Avg Tokens | Latency (s) | Effective GPU Wait |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(1a) 3-Bit Quantized GPU (Standard CoT)** | 0 | 0 MB | -2.2023 | 9.045 | 10.0% (1/10) | 160.0 | 5.31s | 0.000 ms |
| **(1b) 3-Bit Quantized GPU (Caveman Unbridged)** | 0 | 0 MB | -2.2023 | 9.045 | 0.0% (0/10) | 228.0 | 7.80s | 0.000 ms |
| **(1c) 3-Bit GPU + FP16 RAM Bridge ($\mathbf{W}=\mathbf{I}$, Ours)** | **0** | **0 MB** | **-1.7500** | **5.754** | **60.0% (6/10)** | **114.8** | **4.13s** | **0.012 ms** |
| *(1d) Unquantized Crisp FP16 GPU Upper Bound* | *0* | *0 MB* | *-1.1585* | *3.185* | *70.0% (7/10)* | *111.1* | *3.94s* | *0.000 ms* |

---

### 2. Methodology II: Cross-Scale Subspace Translator (`1.5B GPU` $\leftrightarrow$ `7B System RAM`)

Bridging distinct model scales (`Qwen2.5-1.5B` $d=1536$ on GPU $\leftrightarrow$ `Qwen2.5-7B` $d=3584$ middle layers `L12..19` in System RAM):

* **Intra-Layer Mixed-Precision Split (`1% FP16 Outliers + 99% Packed INT4 Bulk`):** Pre-quantization activation outlier masking (`index_fill_`) shrinks the 7B System-RAM slice by **3.48×** (`2816 MB -> 808.15 MB`) and accelerates CPU execution by **2.69×** (`125.0 ms -> 45.93 ms` for 5 routed layers) while preserving **0.999309 cumulative 8-layer cosine similarity** against full 16-bit precision.
* **500-Prompt Answer-Free SVD Alignment (`0 QA Steps`):** Solving closed-form Ridge-Procrustes SVD across 500 unlabeled prompts (`2,000` paired latent states; `450` train / `50` held-out) expands $\mathbf{W}_{\text{up}}$ and $\mathbf{W}_{\text{down}}$ to **full numerical rank (`1536 / 1536`)**, achieving **0.9903 held-out cosine similarity** (`0.9908` train, zero overfitting gap).

| Cross-Scale Configuration (`1.5B GPU + 7B Split RAM`) | QA Steps | 7B RAM Size | Active RAM Layers | GPU Wait (ms) | Avg Tokens | Latency (s) | Solved (`P01–P10`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(2a) 1.5B Standard CoT (GPU Only)** | 0 | 0 MB | 0.0 / 8 | 0.000 ms | 106.3 | 3.55s | 5 / 10 (50%) |
| **(2b) 1.5B Caveman + Superposition (GPU Only)** | 0 | 0 MB | 0.0 / 8 | 0.000 ms | 111.1 | 3.94s | 7 / 10 (70%) |
| **(2c) Answer-Free 500p SVD Bridge (Union of $\mathbf{H}_{\text{out}}$ & $\mathbf{Y}_{\text{disp}}$)** | **0** | **808 MB** | **7.9 / 8** | **0.008 ms** | **101.5** | **3.67s** | **7 / 10 (70%)** |
| **(2d) Disjoint External OOD Bridge (10 External Pairs)** | Ext. | 808 MB | 5.0 / 8 | 0.009 ms | 96.4 | 3.12s | 6 / 10 (60%) |
| **(2e) Task-Calibrated Subconscious Bridge** | **Calib.** | **808 MB** | **5.0 / 8** | **0.009 ms** | **74.1** | **2.88s** | **10 / 10 (100%)** |

---

## 🛠️ Repository Structure

```text
subconscious-engine/
├── src/
│   ├── anchored_superposition.py  # Hybrid Caveman + Soft-Token Superposition & </think> transition gate
│   ├── latent_bridge.py           # 4-Segment Funnel, W=I Precision Bridge, Cross-Scale SVD Bridge
│   ├── mmap_subconscious.py       # IntraLayerSplitQwen2MLP (1% FP16 Outlier + 99% Packed INT4 Bulk)
│   ├── ppo_router.py              # Dynamic Entropy Router & representation velocity early exit
│   └── trainer.py                 # Closed-form 500-prompt Ridge-Procrustes SVD & domain calibration
├── benchmarks/
│   ├── run_ablation.py            # End-to-end benchmark suite across P01-P10
│   └── results/                   # Full per-problem ablation_results.json traces
├── tests/
│   ├── test_vram_budget.py        # GPU VRAM ceiling tripwire (< 5.20 GB)
│   └── verify_pipeline.py         # End-to-end shape, zero-corruption init, and router tripwires
├── paper1_results.json            # Unified Paper #1 experimental tables and microbenchmark metrics
└── ablation_run15.log             # Complete L4 GPU + System RAM benchmark execution log
```

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -e .

# Run local shape & zero-corruption initialization tripwires
python -m src.latent_bridge
python -m tests.verify_pipeline

# Execute the Paper #1 benchmark suite (Methodology I & Methodology II)
python -m benchmarks.run_ablation
```

---

## 📖 Citation

```bibtex
@article{solanki2026subconscious,
  title={Subconscious Latent Offloading: Single-Shot System-RAM Steering Across Quantized and Cross-Scale Language Models},
  author={Solanki, Pratham},
  year={2026},
  month={November},
  note={Preprint}
}
```

---

## 📬 Contact

**Pratham Solanki** — [prathamsolanki.p28@gmail.com](mailto:prathamsolanki.p28@gmail.com)
