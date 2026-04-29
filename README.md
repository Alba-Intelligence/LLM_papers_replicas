# OpenMythos workspace

This repository is now a **multi-package Julia workspace** for model ports derived from the Python reference in `reference/OpenMythos`.

## Workspace layout

- `OpenMythos.jl/` - the existing recurrent OpenMythos Julia package
- `DeepSeekv4.jl/` - the new DeepSeek V4 Julia package
- `TransformerCore.jl/` - reusable shared tensor, normalization, and RoPE primitives
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

The common documentation remains under `docs/wiki/`.
