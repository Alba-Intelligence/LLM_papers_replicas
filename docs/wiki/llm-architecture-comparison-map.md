# Big LLM architecture comparison map

## Purpose

This page records the repository's first broad architecture inventory based on:

- `reference/private/The Big LLM Architecture Comparison.pdf`
- the corresponding article at https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- `reference/private/Big-LLM-Architecture-models.yml`

This is now the workspace-level memory for the broader LLM replication program.

## Scope rules

1. **Implement only the architectures explicitly covered in the PDF/article.**
   The YAML contains many more models than the PDF; those extra entries are useful as a registry, but they are not implementation scope by default.
2. **Use the PDF/article to decide inclusion, grouping, and article order.**
3. **Use the YAML as the structured authority registry** for representative model names, config links, tech-report links, and quick metadata.
4. **Use cited papers, model cards, config files, and reference code** to resolve implementation details once a family is actively being ported.
5. **Prefer clear Julia implementations using existing Julia libraries first.** Shared logic should move into reusable packages only after a real second use appears.

## Already reusable in the workspace

The repo already has a meaningful shared base for the comparison-driven expansion:

- `TransformerCore.jl`
  - feature-last tensor helpers
  - token embeddings / linear heads
  - `RMSNorm`
  - RoPE helpers
  - Lux-native training/checkpoint utilities
  - cache-envelope and growable cache-buffer infrastructure
- `TextDataCore.jl`
  - GPT/tiktoken-style BPE tokenizers
  - local parquet text loading
  - next-token batch construction
- `OpenMythos.jl`
  - GQA and MLA attention variants
  - dense and routed/shared-expert FFN surfaces
  - recurrent decoder scaffolding
- `DeepSeekv4.jl`
  - MoE routing variants
  - MTP heads
  - generation/runtime scaffolding for a non-recurrent decoder family
- `OLMo.jl`
  - dense decoder stack with QK-Norm MHA
  - inside-residual post-norm block layout
  - cache-aware generation and a tiny full-model bootstrap trainer

That means the comparison effort is **not** starting from zero. The existing code already covers several recurring motifs: GQA, MLA, MoE, shared experts, RoPE, RMSNorm, MTP-adjacent training surfaces, and reusable training/runtime infrastructure.

## PDF-covered architecture families

The table below records the article-covered families in the order they appear conceptually in the current article snapshot.

| PDF family                    | Representative authority entry / entries                | Distinguishing motifs to replicate                                                                                                  | Likely Julia family home                                                                                          |
| ----------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| DeepSeek V3 / R1              | `DeepSeek V3`, `DeepSeek R1`                            | MLA, sparse MoE, dense prefix before MoE, shared expert                                                                             | broader DeepSeek-family package; current `DeepSeekv4.jl` is the closest existing nucleus but is V4-specific today |
| OLMo 2                        | `OLMo 2 7B`                                             | dense decoder, MHA, QK-Norm, inside-residual post-norm RMSNorm placement                                                            | new OLMo-family package                                                                                           |
| Gemma 3                       | `Gemma 3 27B`                                           | dense decoder, GQA, 5:1 sliding-window/global schedule, dual pre/post RMSNorm around major blocks                                   | new Gemma-family package                                                                                          |
| Gemma 3n                      | article section plus Google blog authority              | Per-Layer Embeddings, MatFormer slicing, small-device memory/runtime strategy                                                       | Gemma-family package with an efficiency/runtime branch                                                            |
| Mistral Small 3.1             | `Mistral Small 3.1 24B`                                 | latency-oriented dense GQA decoder, classic stack, no active sliding-window path in released config                                 | new Mistral-family package                                                                                        |
| Llama 4                       | `Llama 4 Maverick`                                      | GQA MoE, alternating dense and MoE blocks, fewer/larger experts than DeepSeek                                                       | new Llama-family package                                                                                          |
| Qwen3 dense + MoE             | `Qwen3 0.6B`, `Qwen3 235B-A22B`                         | dense and sparse variants of one family, GQA, QK-Norm on large MoE path, no shared expert in flagship MoE                           | new Qwen-family package                                                                                           |
| SmolLM3                       | `SmolLM3 3B`                                            | dense decoder with periodic NoPE layers                                                                                             | new SmolLM-family package or dense-decoder teaching package if scoped narrowly                                    |
| Kimi K2                       | `Kimi K2`                                               | DeepSeek-style MLA + MoE scaled up, more experts, DeepSeek-like overall template                                                    | likely Kimi-family package; heavily reuses DeepSeek-style shared abstractions                                     |
| gpt-oss                       | `GPT-OSS 20B`, `GPT-OSS 120B`                           | wide-and-shallow sparse MoE, alternating local/global GQA, attention bias, attention sinks                                          | new gpt-oss/OpenAI-open-weights family package                                                                    |
| Grok 2.5                      | `Grok 2.5 270B`                                         | sparse MoE with few large experts and an always-on SwiGLU path behaving like a shared expert                                        | new xAI/Grok family package if prioritized                                                                        |
| GLM-4.5                       | `GLM-4.5 355B`                                          | GQA + QK-Norm, 3 dense layers before MoE, shared expert                                                                             | new GLM-family package                                                                                            |
| Qwen3-Next / Qwen3-Coder-Next | `Qwen3 Next 80B-A3B`                                    | 3:1 Gated DeltaNet / Gated Attention schedule, many experts, shared expert, native MTP/speculative-decoding story                   | Qwen-family package as a major variant                                                                            |
| MiniMax-M2                    | `MiniMax M2 230B`                                       | sparse MoE, per-layer/per-head QK-Norm, partial RoPE, very sparse routing                                                           | new MiniMax-family package                                                                                        |
| Kimi Linear                   | `Kimi Linear 48B-A3B`                                   | 3:1 Kimi Delta Attention / MLA hybrid, NoPE in MLA layers, channel-wise gating                                                      | Kimi-family package as a major variant                                                                            |
| OLMo 3                        | `OLMo 3 7B`, `OLMo 3 32B`                               | OLMo-style post-norm, sliding-window/global schedule, MHA at 7B and GQA at 32B, YaRN on global layers only                          | OLMo-family package                                                                                               |
| DeepSeek V3.2                 | `DeepSeek V3.2`                                         | DeepSeek V3 lineage plus sparse attention on top of MLA + MoE                                                                       | broader DeepSeek-family package                                                                                   |
| Mistral 3 Large               | `Mistral Large 3`                                       | DeepSeek-V3-like MLA + MoE template with different expert granularity                                                               | Mistral-family package with DeepSeek-style shared abstractions underneath                                         |
| Nemotron 3 Nano / Super       | `Nemotron 3 Nano 30B-A3B`, `Nemotron 3 Super 120B-A12B` | Mamba-2 / Transformer hybrid, MoE, sparse attention usage, latent MoE, shared-weight MTP on Super                                   | new Nemotron-family package                                                                                       |
| Xiaomi MiMo-V2-Flash          | `Xiaomi MiMo-V2-Flash 309B`                             | sparse MoE, aggressive 5:1 sliding-window/global schedule, very small local window, MTP                                             | new MiMo-family package                                                                                           |
| Arcee Trinity Large           | `Arcee AI Trinity Large 400B`                           | gated GQA, 3:1 sliding-window/global schedule, QK-Norm, NoPE on global layers, depth-scaled sandwich norm, coarse DeepSeek-like MoE | new Trinity-family package                                                                                        |
| GLM-5                         | `GLM-5 744B`                                            | GLM-4.5 lineage upgraded to MLA plus DeepSeek Sparse Attention                                                                      | GLM-family package                                                                                                |
| Gemma 4 dense + MoE           | `Gemma 4 31B`, `Gemma 4 26B-A4B`                        | Gemma 3-style 5:1 local/global schedule, unusual pre/post norm, key reuse on global layers, partial RoPE, optional MoE sibling      | Gemma-family package                                                                                              |

## Reusable abstraction fronts implied by the PDF

These are the clearest cross-family seams. They should become shared packages only when at least two concrete family implementations use them stably.

| Shared front                       | Article families that need it                                                       | What should eventually become reusable                                                                                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Decoder scaffold + layer schedules | almost all article-covered families                                                 | decoder block stack builders, dense-prefix policies, alternating dense/MoE layouts, local/global layer schedules, hybrid full/linear schedules |
| Attention backends                 | DeepSeek, Gemma, Mistral, Llama, Qwen, GLM, MiniMax, Trinity, gpt-oss               | MHA, GQA, MLA, sliding-window attention, sparse attention, gated attention, key reuse, attention bias, attention sinks                         |
| Positional encoding policies       | DeepSeek, Gemma, SmolLM, Qwen3-Next, Kimi Linear, MiniMax, OLMo 3, Trinity          | RoPE, partial RoPE, NoPE, YaRN wrappers, per-layer positional-policy schedules                                                                 |
| Normalization policies             | OLMo 2/3, Gemma 3/4, Qwen, MiniMax, Trinity                                         | pre-norm, post-norm, dual pre/post norm, sandwich norm, QK-Norm, per-head/per-layer QK-Norm                                                    |
| Sparse FFN / MoE blocks            | DeepSeek, Llama 4, Qwen MoE, gpt-oss, Grok, GLM, MiniMax, Trinity, Xiaomi, Nemotron | routed experts, shared experts, dense-first-then-MoE policies, coarse vs many-small experts, latent experts                                    |
| Linear/state-space sequence blocks | Qwen3-Next, Kimi Linear, Nemotron 3                                                 | Gated DeltaNet, Kimi Delta Attention, Mamba-2-style sequence blocks, full/linear hybrid schedules                                              |
| Auxiliary decoding heads           | DeepSeek lineage, Qwen3-Next, Nemotron 3 Super, Xiaomi                              | MTP heads, shared-weight draft heads, speculative-decoding helpers                                                                             |
| Long-context runtime support       | MLA/sparse/sliding-window/linear families                                           | cache policy selection, latent KV caches, windowed caches, sparse caches, hybrid cache/state management                                        |
| Device/runtime memory tricks       | Gemma 3n, Nemotron Super                                                            | PLE-style streamed parameters, latent expert staging, runtime-only memory policies                                                             |

## What is still missing in the repo

The main missing components relative to the PDF map are:

- sliding-window attention as a first-class backend
- sparse attention variants
- NoPE and partial-RoPE policies
- QK-Norm variants beyond the current simple form
- gated attention and attention-sink mechanics
- dense/local/global layer scheduling helpers
- Gated DeltaNet and Kimi Delta Attention
- Mamba-2-style state-space blocks
- latent-space MoE
- key-reuse attention policies
- PLE / MatFormer-style runtime-efficient parameter slicing
- family packages beyond the current OpenMythos, DeepSeek, and first OLMo slices

## Packaging guidance for the next implementation phase

1. Keep `TransformerCore.jl` focused on low-level, architecture-agnostic math, training, and cache primitives.
2. Keep `TextDataCore.jl` focused on tokenizer/text-data helpers.
3. Add **new family packages** when a PDF family has a real internal identity across multiple variants:
   - Gemma family
   - OLMo family
   - Qwen family
   - GLM family
   - Kimi family
   - Mistral family
   - Nemotron family
   - and so on as work is prioritized
4. Extract **new shared packages** only when the same abstraction has stabilized across at least two family packages.
5. Treat current `DeepSeekv4.jl` as a useful implementation nucleus, but do not force all DeepSeek-lineage work into a V4-only shape if a broader DeepSeek-family split becomes cleaner.

## Initial implementation order for the first three additional families

The current recommended sequence is:

1. **`OLMo.jl` first** - landed as the first additional family slice
2. **`Gemma.jl` second**
3. **`Qwen.jl` third**

### Why this order

#### 1. `OLMo.jl` first

`OLMo.jl` is the cleanest next dense-decoder family to add because it introduces:

- a non-recurrent decoder family without MoE pressure,
- package-local MHA plus QK-Norm,
- the OLMo-specific inside-residual post-norm layout,
- a strong clarity-first baseline for future decoder families.

This makes it the best first family for proving a reusable non-recurrent decoder shell **without** forcing sliding-window, MoE, or linear-attention abstractions too early.

The first `OLMo.jl` slice should target **OLMo 2 first**, with OLMo 3 noted as the natural follow-on inside the same package.

#### 2. `Gemma.jl` second

`Gemma.jl` is the next best family because it adds a second dense-decoder implementation while introducing the next reusable front:

- GQA as the main attention path,
- explicit **5:1 local/global sliding-window scheduling**,
- the Gemma-style **dual pre/post norm** layout,
- later room for Gemma 3n and Gemma 4 variants inside one family package.

By doing Gemma second, the repository gets a genuine second user for a decoder scaffold while keeping sliding-window scheduling package-local until it has proved itself in real code.

#### 3. `Qwen.jl` third

`Qwen.jl` should come after OLMo and Gemma because it is the first family where it is clearly worth representing **both dense and MoE variants** in one package.

That timing matters because by then the repo should already have:

- one stable dense non-recurrent baseline (`OLMo.jl`),
- one dense decoder with local/global scheduling (`Gemma.jl`),
- enough experience to decide which decoder-shell pieces are actually reusable.

Qwen is the right point to test whether shared decoder scaffolding is real and whether sparse FFN / MoE code should eventually be extracted.

## Package plan for those first three families

| Order | Family package | First model slice | Keep package-local at first | Candidate later extraction |
| --- | --- | --- | --- | --- |
| 1 | `OLMo.jl` | OLMo 2 tiny dense decoder | post-norm block policy, package-local QK-Norm, MHA stack assembly | non-recurrent decoder shell pieces only if Gemma confirms them |
| 2 | `Gemma.jl` | Gemma 3 tiny dense decoder | sliding-window attention/cache logic, 5:1 local/global scheduler, dual pre/post norm policy | decoder shell and maybe layer-schedule interfaces after OLMo + Gemma |
| 3 | `Qwen.jl` | Qwen3 tiny dense preset plus tiny MoE preset | Qwen-specific QK-Norm details, MoE router/container choices | sparse FFN / MoE helpers only after direct comparison with DeepSeek/Qwen |

## Delayed shared-package checkpoints

### After `OLMo.jl` and `Gemma.jl`

If the overlap is real, extract a **shared non-recurrent decoder scaffold** package.

Tentative name:

- `DecoderCore.jl`

Expected scope:

- decoder stack assembly,
- per-layer dispatch hooks,
- generic causal decoder/generation shell,
- reusable layer-schedule plumbing.

This package should **not** absorb Gemma-specific sliding-window logic, Gemma's norm placement, or OLMo's post-norm policy until those APIs stabilize further.

### After `Qwen.jl`

Only then evaluate whether a shared sparse FFN / MoE package is justified.

Tentative name:

- `SparseFFNCore.jl`

Expected scope only if convergence is real:

- expert containers,
- routed-expert selection plumbing,
- shared-expert hooks,
- MoE aggregation helpers.

This extraction should be justified by direct side-by-side comparison between `Qwen.jl` and the existing DeepSeek-family code. If the APIs are still family-shaped, leave the MoE code package-local.

## What this order deliberately defers

This sequence deliberately defers:

- broader DeepSeek-family repackaging beyond the current `DeepSeekv4.jl` nucleus,
- Mistral and Llama family work,
- GLM and Kimi family work,
- linear/state-space hybrids such as Qwen3-Next, Kimi Linear, and Nemotron,
- Gemma 3n runtime-specific work,
- production-scale runtime optimization work.

The goal is to earn the abstractions in a clarity-first order rather than chase the flashiest architectures first.

## Immediate repository-level next step

With the first `OLMo.jl` slice now landed, the repository should keep this sequence:

1. map the next target family (`Gemma.jl`) to its PDF/YAML authority entries,
2. keep sliding-window scheduling and dual pre/post norm package-local on that first Gemma use,
3. compare `Gemma.jl` against `OLMo.jl` before attempting any `DecoderCore.jl` extraction,
4. only after that, add `Qwen.jl` as the first dense+MoE family package,
5. update this page and the broader wiki before each commit.
