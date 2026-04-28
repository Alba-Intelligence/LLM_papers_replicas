# Architecture

## What the Python reference implements

The main implementation lives in `reference/OpenMythos/open_mythos/main.py` and builds a decoder-only language model around a looped middle block.

```text
Input IDs
  -> token embedding
  -> Prelude: standard transformer blocks, run once
  -> Recurrent Block: one transformer block reused for multiple loop steps
  -> Coda: standard transformer blocks, run once
  -> RMSNorm
  -> LM head
```

The key invariant is that the encoded prelude output `e` is frozen and injected into every recurrent step. The recurrent hidden state is not left to drift on its own.

## Main components

### 1. `MythosConfig`

`MythosConfig` carries nearly all architecture choices:

- model width and sequence limits,
- recurrent loop count,
- attention backend (`"gqa"` or `"mla"`),
- MoE routing dimensions,
- ACT threshold,
- LoRA rank,
- RoPE settings.

For the Julia port, this should become a single typed config object rather than many loosely coupled keyword arguments.

### 2. Prelude and Coda

Prelude and Coda are ordinary pre-norm transformer stacks that run once. They use:

- RMSNorm,
- the selected attention backend,
- dense SwiGLU-style FFNs.

They are important structurally, but not novel by themselves.

### 3. Recurrent Block

The recurrent block is what makes OpenMythos distinct.

Per loop iteration, the Python code does:

1. inject a loop-index embedding into the hidden state,
2. combine hidden state with the frozen encoded input `e`,
3. apply one transformer block with MoE FFN,
4. add a depth-wise LoRA delta,
5. update the state with an LTI-stable injection rule,
6. compute ACT halting probabilities and accumulate a weighted output.

This means the Julia port should not start from "generic transformer.jl" and bolt recurrence on later. The recurrence logic is the model.

### 4. Attention backends

The reference supports two attention modes:

| Backend | Python class | Role in the port |
| --- | --- | --- |
| GQA | `GQAttention` | Simpler baseline path; good first Julia attention target |
| MLA | `MLAttention` | Higher-priority for parity because it is the default path in the reference |

Both backends use RoPE and KV caching, but the cache representation differs:

- GQA caches full K and V tensors with fewer KV heads than Q heads.
- MLA caches a compressed latent representation and reconstructs parts of K and V on demand.

## Why the tests matter

The strongest cues for a Julia port are not the training scripts. They are the invariants encoded in `tests/test_main.py`:

- RMSNorm shape and RMS properties,
- RoPE shape, norm preservation, and relative-position behavior,
- cache behavior,
- halting logic,
- LoRA depth handling,
- recurrent block output shape and stability assumptions.

Those tests describe the minimum correctness bar for the Julia code.

## What is primary vs secondary

### Primary port target

- `open_mythos/main.py`
- `open_mythos/tokenizer.py`
- `open_mythos/variants.py`
- `tests/test_main.py`
- `tests/test_tokenizer.py`
- `docs/open_mythos.md`

### Secondary target

- `training/3b_fine_web_edu.py`
- `docs/datasets.md`

### Experimental / defer until core parity exists

- `open_mythos/moda.py`
- `examples/moda_example.py`
- benchmark scripts in `tests/`

