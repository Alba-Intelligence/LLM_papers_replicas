# OpenMythos workspace

This repository is now a **multi-package Julia workspace** for model ports derived from the Python reference in `reference/OpenMythos`.

## Workspace layout

- `OpenMythos.jl/` - the existing recurrent OpenMythos Julia package
- `DeepSeekv4.jl/` - the new DeepSeek V4 Julia package
- `TransformerCore.jl/` - reusable shared tensor, normalization, RoPE, and training utilities
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
