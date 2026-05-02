# OpenMythos API

```@meta
CurrentModule = OpenMythos
```

## Package overview

`OpenMythos.jl` is the Julia implementation of the recurrent-depth OpenMythos
architecture. It combines:

- a prelude/recurrent/coda transformer layout,
- ACT-based halting,
- LTI input injection,
- loop-index embeddings,
- depth-wise LoRA adapters,
- MoE feed-forward routing,
- cache-aware generation and bootstrap training helpers.

## Core model

`OpenMythos` is the package's full recurrent-depth model type. It owns the token
embedding table, the prelude transformer stack, the shared recurrent block, the
coda stack, the final normalization, and the LM head. Construct it with:

```julia
model = OpenMythos(cfg; rng=Random.default_rng(), T=Float32)
```

The main forward/runtime entrypoints around that type are documented below:

## Exported API

```@docs
MythosConfig
loop_index_embedding
GQAttention
MLAttention
Expert
MoEFFN
TransformerBlock
RecurrentBlock
LoRAAdapter
LTIInjection
get_A
ACTHalting
generate
DEFAULT_MODEL_ID
MythosTokenizer
vocab_size
tokenize
detokenize
bootstrap_training_config
bootstrap_full_model_training_config
fineweb_edu_batches
LuxHeadOnlyOpenMythos
HeadOnlyTrainerState
FullModelTrainerState
head_only_logits
head_only_loss
full_model_logits
full_model_loss
train_head_only_step!
train_full_model_step!
train_head_only!
train_full_model!
chunked_prefill
save_head_only_checkpoint
load_head_only_checkpoint
save_full_model_checkpoint
load_full_model_checkpoint
mythos_1b
mythos_3b
mythos_10b
mythos_50b
mythos_100b
mythos_500b
mythos_1t
```
