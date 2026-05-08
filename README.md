# Julia research replication workspace

This repository is a **general-purpose multi-package Julia workspace** for reimplementing, studying, and extending research ideas.

It is not limited to one paper or even one subfield. The repo currently contains:

- **LLM architecture ports and training experiments**
- **graph / hypergraph utilities and replicas**
- **research notebooks and exploratory packages**
- **shared infrastructure for reusable Julia implementations**

## Current main workspace themes

### Language-model replication workspace

The most actively documented subworkspace currently centers on model-family ports:

- `OpenMythos.jl/`
  - recurrent-depth OpenMythos implementation in Julia
  - covers `Prelude -> Recurrent Block -> Coda`, GQA/MLA attention, MoE FFNs, ACT halting, loop-index embeddings, depth-wise LoRA, LTI-stable input injection, cache reuse, tokenizer parity, and staged Lux-oriented training work
- `DeepSeekv4.jl/`
  - architecture-first DeepSeek V4 implementation in Julia
  - covers CSA/HCA hybrid attention, mHC residual mixing, routed/shared/hash MoE paths, optional Engram conditional memory, MTP heads, cached generation reuse, and tiny-config Lux bootstrap training
- `TransformerCore.jl/`
  - shared infrastructure for model-family ports
  - covers feature-last tensor helpers, embedding/sampling helpers, `RMSNorm`, RoPE, shared Lux-native training/checkpoint utilities, growable cache buffers, and serialized KV-cache envelopes
- `TextDataCore.jl/`
  - shared Julia-native tokenizer and local text-data helpers for model packages
  - covers GPT/tiktoken-style BPE tokenizers, vocabulary surface extraction, and parquet-backed next-token batch loading
- `docs/wiki/`
  - shared architecture and implementation notes for the model-port workspace

### Other research/utility packages

The repository also contains other Julia packages and experiments at the root, including graph-oriented and hypergraph-oriented work such as:

- `GraphAnalysis.jl/`
- `GraphGeneration.jl/`
- `GraphTools.jl/`
- `GraphUtils.jl/`
- `HypergraphReasoning/`

These packages are part of the broader research-replication workspace even when the wiki is currently focused more heavily on the LLM/model-port side.

## Quick commands

### OpenMythos tests

```bash
cd OpenMythos.jl
OPENMYTHOS_TEST_TOKENIZER_MODEL_ID=gpt2 julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### DeepSeek V4 tests

```bash
cd DeepSeekv4.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### TransformerCore tests

```bash
cd TransformerCore.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### TextDataCore tests

```bash
cd TextDataCore.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### DeepSeek V4 tiny training smoke run

```bash
cd DeepSeekv4.jl
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

## Documentation

- model-port documentation currently lives under `docs/wiki/`
- package-local READMEs provide more detail for each subproject
- notebooks and scripts under the repo root capture more exploratory work
