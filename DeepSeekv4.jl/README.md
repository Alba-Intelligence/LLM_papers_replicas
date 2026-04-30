# DeepSeekV4.jl

Architecture-first Julia implementation workspace for DeepSeek V4.

This package currently targets:

- tiny-config forward passes,
- readable CSA/HCA reference attention,
- manifold-constrained hyper-connections,
- DeepSeek-style MoE routing surfaces,
- MTP output heads and generation smoke paths,
- a Lux-backed head-only bootstrap training surface for tiny configs.

## Quickstart

### Run the test suite

```bash
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### Bootstrap training smoke run

```bash
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

This script currently builds local byte-encoded batches for a simple trainable smoke path.

The shared wiki for this repository remains in `../docs/wiki/`.
