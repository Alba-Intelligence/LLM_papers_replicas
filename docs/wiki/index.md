# OpenMythos Julia Wiki

This wiki is the narrative spine for the Julia reimplementation of the Python reference in `reference/OpenMythos`.

It is intentionally concept-first: the goal is to explain what the Python project is doing, what parts matter for the port, and how the Julia codebase should grow without forcing readers to reverse-engineer `main.py` from scratch.

## Current repository state

- The root repository now contains a Julia package scaffold with `Project.toml`, `src/`, `test/`, and a resolved `Manifest.toml`.
- The authoritative implementation today is the Python reference at `reference/OpenMythos`.
- The implemented Julia slice now covers the primitive layer, the main model stack, named presets, a tokenizer bridge, an optional Python-vs-Julia parity harness, a Lux-backed bootstrap training/data path, and package-level usage docs.
- The tracked roadmap items are now in place; future work focuses on deeper training/runtime capability rather than missing repository basics.

## Reading order

1. [Usage](usage.md) - how to actually run the current Julia package.
2. [Architecture](architecture.md) - the model we are actually porting.
3. [Python reference map](python-reference-map.md) - which Python files matter and how they translate into Julia work.
4. [References](references.md) - papers, datasets, and implementation references already cited by the Python project.
5. [Julia reimplementation plan](julia-reimplementation-plan.md) - target module layout, libraries, and implementation phases.

## Big picture

The main OpenMythos path is a Recurrent-Depth Transformer:

```text
tokens
  -> embedding
  -> prelude blocks
  -> recurrent block looped T times
  -> coda blocks
  -> norm + lm head
```

The recurrent block is the center of gravity. It combines:

- shared transformer weights across loop iterations,
- stable input injection,
- ACT halting,
- depth-wise LoRA,
- MoE feed-forward routing,
- switchable GQA or MLA attention.

That combination is what the Julia port should preserve first. Everything else is secondary.

## Scope guidance

- Treat `reference/OpenMythos/open_mythos/main.py` as the behavioral core.
- Treat `reference/OpenMythos/docs/open_mythos.md` as the clearest architecture narrative.
- Treat `reference/OpenMythos/open_mythos/moda.py` as a secondary experimental branch, not the first porting target.
- Port invariants before porting scale.

## Maintenance rule

After a task is complete, update the relevant wiki pages if the task changed implementation status, architecture understanding, or workflow guidance, then record the work in git with a commit.

## Training status

The Julia replica now has a real bootstrap training path:

- token chunking and next-token batch construction,
- warmup + cosine learning-rate scheduling,
- resumable checkpointing,
- a small Julia `train_3b_fineweb_edu.jl` entrypoint,
- local-text smoke training and an optional FineWeb-Edu Python streaming bridge,
- a `LuxHeadOnlyOpenMythos` layer backed by `Optimisers.AdamW` and `NNlib.logsoftmax`.

That path is intentionally limited to **head-only** optimization for now.

## Future work

The current roadmap items are complete. The main remaining directions are:

- full-model gradient-based training beyond the head-only Lux layer,
- distributed training/runtime behavior,
- deeper performance work,
- optional future documentation/API expansion as those land.
