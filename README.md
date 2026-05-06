# OpenMythos workspace

This repository is now a **multi-package Julia workspace** for model ports derived from the Python reference in `reference/OpenMythos`.

## Workspace layout

- `OpenMythos.jl/` - the existing recurrent OpenMythos Julia package
  - Implements the recurrent-depth stack: `Prelude -> Recurrent Block -> Coda`, switchable GQA/MLA attention, MoE FFNs, ACT halting, loop-index embeddings, depth-wise LoRA, LTI-stable input injection, tokenizer parity, cache reuse, and Lux-backed bootstrap training.
  - ArXiv references: [Loop, Think, & Generalize / Recurrent-Depth Transformer](https://arxiv.org/pdf/2604.07822), [Parcae / Scaling Laws for Stable Looped Language Models](https://arxiv.org/abs/2604.12946), [Universal Transformers](https://arxiv.org/pdf/1807.03819), [DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434), [GQA](https://arxiv.org/abs/2305.13245), [DeepSeekMoE](https://arxiv.org/abs/2401.06066), [Relaxed Recursive Transformers](https://arxiv.org/pdf/2410.20672), [RMSNorm](https://arxiv.org/abs/1910.07467), [RoPE](https://arxiv.org/abs/2104.09864), and [ACT](https://arxiv.org/abs/1603.08983).
- `DeepSeekv4.jl/` - the new DeepSeek V4 Julia package
  - Implements architecture-first DeepSeek V4 components: CSA/HCA hybrid attention, mHC residual mixing, routed/shared/hash MoE paths, optional Engram conditional memory, MTP heads, cached generation reuse, and tiny-config Lux bootstrap trainers.
  - References: the official [DeepSeek V4 technical note](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), plus the arXiv lineage used in the Julia package: [DeepSeek-V3](https://arxiv.org/html/2412.19437v1), [Engram](https://arxiv.org/abs/2601.07372), and [mHC](https://arxiv.org/abs/2512.24880).
- `TransformerCore.jl/` - reusable shared tensor, normalization, RoPE, and training utilities
  - Implements architecture-agnostic feature-last tensor helpers, embedding/sampling helpers, `RMSNorm`, RoPE precomputation and application, warmup-cosine schedules, token chunking/batching, checkpoint discovery, growable cache buffers, and serialized KV-cache envelopes.
  - ArXiv references: [RMSNorm](https://arxiv.org/abs/1910.07467) and [RoPE](https://arxiv.org/abs/2104.09864).
- `docs/wiki/` - shared architecture and workspace documentation
- `reference/OpenMythos/` - upstream Python reference

## Quick commands

### OpenMythos

```bash
cd OpenMythos.jl
OPENMYTHOS_TEST_TOKENIZER_MODEL_ID=gpt2 julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### DeepSeek V4

```bash
cd DeepSeekv4.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### DeepSeek V4 tiny training smoke run

```bash
cd DeepSeekv4.jl
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

The common documentation remains under `docs/wiki/`.
