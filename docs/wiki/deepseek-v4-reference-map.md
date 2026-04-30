# DeepSeek V4 reference map

This page maps the DeepSeek V4 source materials used for this Julia workspace to the current package layout.

Unlike OpenMythos, there is no local Python reference tree in this repository for DeepSeek V4. The source of truth is therefore the external release material plus the current Julia implementation choices.

## Source-of-truth order

1. DeepSeek V4 technical note: `DeepSeek_V4.pdf`
2. DeepSeek V4 introduction video
3. `DeepSeekv4.jl/src/config.jl`
4. `DeepSeekv4.jl/src/attention.jl`
5. `DeepSeekv4.jl/src/mhc.jl`
6. `DeepSeekv4.jl/src/ffn.jl`
7. `DeepSeekv4.jl/src/deepseek_model.jl`
8. `DeepSeekv4.jl/test/test_model_stack.jl`

## Concept-to-file map

| DeepSeek V4 concept | Current Julia target | Notes |
| --- | --- | --- |
| model hyperparameters and variant metadata | `DeepSeekv4.jl/src/config.jl`, `DeepSeekv4.jl/src/variants.jl` | includes tiny and paper-inspired presets |
| CSA branch | `DeepSeekv4.jl/src/attention.jl` | readable reference implementation |
| HCA branch | `DeepSeekv4.jl/src/attention.jl` | correctness-first path, not fused |
| mHC residual mixing | `DeepSeekv4.jl/src/mhc.jl` | explicit Sinkhorn-style residual mixing |
| routed/shared/hash MoE | `DeepSeekv4.jl/src/ffn.jl` | early hash-routed layers, later regular MoE |
| non-recurrent DeepSeek block stack | `DeepSeekv4.jl/src/deepseek_model.jl` | package-level model assembly |
| MTP outputs | `DeepSeekv4.jl/src/deepseek_model.jl` | exposed through `mtp_logits` |
| generation smoke path | `DeepSeekv4.jl/src/deepseek_model.jl` | small-cache decode path |
| architecture invariants | `DeepSeekv4.jl/test/test_model_stack.jl` | shapes, cache smoke, generation smoke |

## Current interpretation boundaries

The present Julia package intentionally captures:

- CSA + HCA hybrid attention,
- manifold-constrained hyper-connections,
- DeepSeek-style MoE routing surfaces,
- MTP output heads,
- cache-aware tiny-config generation.

It intentionally does **not** yet claim parity for:

- production million-token runtime,
- Muon or hybrid ZeRO training,
- FP4 QAT,
- contextual parallelism,
- release-checkpoint loading.

## Reading order for implementation work

1. start with the technical note for the high-level system picture,
2. use the video to sanity-check terminology and intended emphasis,
3. read `config.jl` and `variants.jl` to see what has been concretized,
4. read `attention.jl` and `mhc.jl` for the model-family-specific mechanics,
5. read `deepseek_model.jl` and the tiny tests to understand the current executable surface.
