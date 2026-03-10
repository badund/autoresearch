# autoresearch (Windows + NVIDIA GPU, Optimized)

> Turn your Windows gaming PC into an autonomous AI researcher — with Triton, `torch.compile`, and tuned hyperparameters.

This fork builds on [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (the Windows port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)) and adds Triton-on-Windows support, `torch.compile`, flex attention, and hyperparameter tuning for consumer NVIDIA GPUs.

## Performance

All benchmarks on **RTX 3090 24GB**, Windows 11 (Driver 581.95, CUDA 13.0), 5-minute training budget, TinyStories dataset.

| Version | val_bpb | MFU | Steps | Throughput | VRAM |
|---------|---------|-----|-------|------------|------|
| jsegov fork (SDPA + eager) | 1.008 | 12.1% | 52 | 91K tok/s | 11.9 GB |
| + `torch.compile` only | 0.883 | 15.1% | 62 | 108K tok/s | 6.2 GB |
| + attention optimizations | 0.611 | 31.4% | 118 | 206K tok/s | 6.1 GB |
| **This fork (fully tuned)** | **0.472** | **32.5%** | **602** | **197K tok/s** | **6.1 GB** |

**53% lower val_bpb**, **2.7x MFU**, **11.6x more gradient steps**, **47% less VRAM** vs the base Windows fork.

### Comparison to jsegov/autoresearch-win-rtx

| | [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) | This fork |
|---|---|---|
| Triton | Not used (explicit non-goal) | `triton-windows 3.5.x` via PyPI |
| `torch.compile` | Disabled | Enabled (`max-autotune-no-cudagraphs`) |
| Attention | SDPA with explicit mask | `is_causal=True` + flex_attention for sliding windows |
| Optimizer kernels | Eager | Compiled (`fullgraph=True`) |
| Batch size | 2^19 (524K tokens) | 98,304 tokens (tuned sweet spot) |
| Newton-Schulz iters | 5 | 3 |
| Weight decay | 0.2 | 0.05 |
| Warmdown ratio | 0.5 | 0.7 |
| Final LR fraction | 0.0 | 0.02 |
| Matrix LR | 0.04 | 0.05 |

## What changed

**Triton + compile pipeline.** The jsegov fork intentionally disabled `torch.compile` and Triton to avoid unofficial Windows stacks. The [`triton-windows`](https://pypi.org/project/triton-windows/) package (v3.5.x) now provides official-quality Triton support on Windows via PyPI, enabling the full compile pipeline.

**Attention optimizations.** Full-context layers use `is_causal=True` (hardware-accelerated causal mask, no memory allocation). Sliding-window layers use `flex_attention` with precomputed block masks, compiled via `torch.compile`.

**Hyperparameter tuning.** Smaller effective batch size (98K tokens vs 524K) yields 5x more gradient steps in the same time budget. Lower weight decay and higher matrix LR suit the undertrained regime (~1.2 tokens per parameter). Longer warmdown with a small final LR floor improves end-of-training convergence.

## Quick start

**Requirements:** Windows 10/11, NVIDIA GPU (10+ GB VRAM), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```powershell
# 1. Install uv (if needed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies (includes triton-windows automatically)
uv sync

# 3. Download data and train tokenizer (one-time)
uv run prepare.py

# 4. Smoke test
uv run train.py --smoke-test

# 5. Full training run (~5 min)
uv run train.py
```

`uv sync` installs `triton-windows>=3.5,<3.6` from PyPI alongside PyTorch 2.9.1 (CUDA 12.8). No manual Triton installation required.

### Environment variables

| Variable | Effect |
|---|---|
| `AUTORESEARCH_AUTOTUNE_REFRESH=1` | Re-run the batch size autotune (recommended after GPU or driver changes) |
| `AUTORESEARCH_DISABLE_AUTOTUNE=1` | Skip autotune, use default batch sizes |
| `AUTORESEARCH_FORCE_COMPILE=0` | Disable `torch.compile` (fallback to eager) |
| `AUTORESEARCH_FORCE_CHECKPOINTING=1` | Force activation checkpointing on |

## Running the agent

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

Point Claude, Codex, or any agent at this repo. The `program.md` file provides the experiment loop instructions. The agent modifies only `train.py`.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Platform support

Same GPU support matrix as [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx):

- **Ampere:** RTX 3060 12GB, 3070, 3080, 3080 Ti, 3090, 3090 Ti
- **Ada:** RTX 4060 Ti 16GB, 4070, 4070 SUPER, 4070 Ti, 4070 Ti SUPER, 4080, 4080 SUPER, 4090
- **Blackwell:** RTX 5060 Ti, 5070, 5070 Ti, 5080, 5090
- Desktop only (no laptop GPUs). 8 GB variants not supported.
- Tested on RTX 3090 24GB, Windows 11.

## Cross-Platform Comparison: MLX vs Windows + Triton

Head-to-head on the **same dataset** (TinyStories), **same model** (50.3M params, depth 8), **same 5-minute training budget**. Eval uses 40 × 524,288 tokens.

### Hardware

| | Apple MLX | Windows + Triton |
|---|---|---|
| Machine | MacBook Pro 16-inch (2023) | Desktop PC |
| Chip / GPU | Apple M2 Max | NVIDIA RTX 3090 |
| Memory | 64 GB unified | 24 GB VRAM |
| Runtime | MLX | PyTorch 2.9 + Triton 3.5 |
| Compile | MLX JIT | `torch.compile` (max-autotune-no-cudagraphs) |

### Results

| Metric | MLX (M2 Max) | Windows + Triton (RTX 3090) | Delta |
|---|---|---|---|
| **val_bpb** | 1.6025 | **0.4713** | **70.6% lower** |
| **tokens trained** | 6.2M | **59.2M** | **9.5x more** |
| **training steps** | 94 | **602** | **6.4x more** |
| **tok/sec** | ~20.5K | **~197K** | **9.6x faster** |
| **peak memory** | 26.7 GB | **6.0 GB** | **4.5x less** |
| training time | 302s | 300s | same budget |
| eval time | ~422s | ~95s | 4.4x faster |
| total wall clock | 724.5s | 394.8s | 1.8x faster |
| parameters | 50.3M | 50.3M | identical |

### Why the gap?

The RTX 3090 processes **9.6x more tokens per second**, which compounds into 6.4x more gradient steps in the same budget. More steps = more learning = lower val_bpb. Key factors:

- **Raw compute:** The RTX 3090 (35.6 TFLOPS FP32 / 71 TFLOPS TF32) dramatically outpaces the M2 Max GPU (~13.6 TFLOPS FP32) for matrix-heavy workloads.
- **Triton + torch.compile** fuse operations and eliminate Python/framework overhead, achieving ~33% MFU vs MLX's less mature compiler.
- **Smaller effective batch size** (98K vs 65K tokens) enables more frequent weight updates.
- **Tuned hyperparameters** (lower weight decay, longer warmdown) improve convergence in the high-step regime.

> **Upstream reference:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch) on H100: val_bpb **0.998** in the same 5-minute budget.

## Notable forks

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — original (Linux/H100)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) — Windows port (SDPA + eager)
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — macOS
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Apple MLX

## License

MIT
