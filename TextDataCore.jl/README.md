# TextDataCore.jl

Small shared Julia package for text-tokenizer and local text-data helpers used by
multiple model packages in this workspace.

Current scope:

- Julia-native GPT/tiktoken-style BPE tokenizers via `BytePairEncoding.jl`
- zero-based encode/decode API for model-training code
- vocabulary surface extraction for token-aware utilities such as Engram lookup building
- local parquet text-column loading via `Parquet2.jl`
- next-token batch construction from parquet shard(s)

This package is intentionally separate from `TransformerCore.jl` so the latter
can stay focused on architecture-agnostic tensor/training/runtime primitives
without taking on tokenizer or dataset-format policy.
