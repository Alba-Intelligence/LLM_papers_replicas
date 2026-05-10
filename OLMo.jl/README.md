# OLMo.jl

`OLMo.jl` is the Julia family package for OLMo models in this workspace.

The first implemented slice targets **OLMo 2** as a dense decoder-only transformer with:

- RoPE,
- MHA,
- QK-Norm,
- SwiGLU,
- the OLMo-specific inside-residual post-norm layout,
- cache-aware generation,
- a tiny full-model bootstrap trainer built on shared checkpoint utilities.

## Current scope

Implemented first:

- `OLMoConfig`
- `OLMoAttention`
- `OLMoMLP`
- `OLMoBlock`
- `OLMoModel`
- `generate`
- `chunked_prefill`
- `OLMoTokenizer` backed by shared `TextDataCore.jl`
- `bootstrap_olmo_training_config`
- `OLMoFullModelTrainerState`
- `train_olmo_step!`
- `train_olmo!`
- `save_olmo_checkpoint`
- `load_olmo_checkpoint`
- `scripts/train_olmo_tiny.jl`

Still deferred:

- exact OLMo tokenizer parity with the extra masking-token extension over `cl100k_base`,
- OLMo 3 sliding-window / mixed local-global scheduling,
- any shared `DecoderCore.jl` extraction before a second concrete dense decoder family justifies it.

## Quick commands

### Tests

```bash
cd OLMo.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### Tiny training smoke run

```bash
cd OLMo.jl
OLMO_TRAIN_TOTAL_STEPS=8 \
OLMO_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_olmo_tiny.jl
```
