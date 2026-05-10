# OLMo reference map

This page maps the OLMo source materials used for this Julia workspace to the current `OLMo.jl` package layout.

The first Julia slice targets **OLMo 2** first, with OLMo 3 treated as the next package-internal follow-on rather than a separate family package.

## Source-of-truth order

1. OLMo 2 paper: https://arxiv.org/pdf/2501.00656
2. OLMo 2 Hugging Face config for the 7B model
3. Hugging Face `modeling_olmo2.py` / `modular_olmo2.py`
4. `OLMo.jl/src/config.jl`
5. `OLMo.jl/src/attention.jl`
6. `OLMo.jl/src/blocks.jl`
7. `OLMo.jl/src/model.jl`
8. `OLMo.jl/src/training.jl`
9. `OLMo.jl/test/test_model_stack.jl`
10. `OLMo.jl/test/test_training.jl`

## Concept-to-file map

| OLMo concept | Current Julia target | Notes |
| --- | --- | --- |
| family hyperparameters | `OLMo.jl/src/config.jl`, `OLMo.jl/src/variants.jl` | includes tiny and published 7B-style presets |
| QK-Norm MHA | `OLMo.jl/src/attention.jl` | first slice keeps MHA package-local |
| SwiGLU feed-forward block | `OLMo.jl/src/blocks.jl` | bias-free dense FFN path |
| inside-residual post-norm decoder layer | `OLMo.jl/src/blocks.jl` | matches the OLMo 2 block ordering |
| full dense decoder stack | `OLMo.jl/src/model.jl` | package-level model assembly |
| cache-aware generation | `OLMo.jl/src/model.jl` | shared outer envelope, OLMo-local inner cache payload |
| bootstrap full-model trainer | `OLMo.jl/src/training.jl` | shared checkpoint layout, package-local loss/update wrapper |
| tokenizer wrapper | `OLMo.jl/src/tokenizer.jl` | currently backed by `cl100k_base` through `TextDataCore.jl` |
| architecture invariants | `OLMo.jl/test/test_model_stack.jl` | forward shape, cache smoke, generation smoke |
| training invariants | `OLMo.jl/test/test_training.jl` | repeated-batch loss reduction, checkpoint roundtrip, final checkpoint write |

## Current interpretation boundaries

The present Julia package intentionally captures:

- dense decoder-only OLMo 2 model assembly,
- RoPE,
- QK-Norm attention,
- OLMo 2 inside-residual post-norm,
- SwiGLU feed-forward blocks,
- shared outer cache-envelope integration,
- a tiny full-model bootstrap trainer and smoke script.

It intentionally does **not** yet claim parity for:

- exact OLMo tokenizer parity with the extra masking-token extension over `cl100k_base`,
- OLMo 3's mixed local/global sliding-window schedule,
- OLMo 3's larger-model GQA path,
- full training-recipe parity with the original infrastructure,
- optimized kernels or large-scale serving.

## Reading order for implementation work

1. start with the OLMo 2 paper for the architecture changes relative to earlier OLMo releases,
2. use the 7B Hugging Face config to lock the concrete published field values,
3. read the Hugging Face `olmo2` modeling code to confirm block ordering and QK-Norm placement,
4. read `config.jl`, `attention.jl`, and `blocks.jl` for the current Julia interpretation,
5. read `model.jl` and the smoke tests to understand the current executable surface.
