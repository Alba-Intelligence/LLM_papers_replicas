# Python Reference Map

This page is the bridge between the current Python project and the future Julia layout.

## Source-of-truth order

1. `reference/OpenMythos/open_mythos/main.py` - ground-truth behavior.
2. `reference/OpenMythos/docs/open_mythos.md` - best prose description of the architecture.
3. `reference/OpenMythos/tests/test_main.py` - executable invariants.
4. `reference/OpenMythos/open_mythos/variants.py` - preset model scales.
5. `reference/OpenMythos/training/3b_fine_web_edu.py` - current training workflow.
6. `reference/OpenMythos/README.md` - useful overview, but more speculative and less implementation-tight.

## File-by-file mapping

| Python path | Purpose | Julia target | Priority |
| --- | --- | --- | --- |
| `open_mythos/main.py` | Core model, layers, recurrent loop, generation | `OpenMythos.jl/src/` core modules and model assembly | Highest |
| `open_mythos/tokenizer.py` | Hugging Face tokenizer wrapper | `OpenMythos.jl/src/tokenizer.jl` or a temporary Python bridge | High |
| `open_mythos/variants.py` | Named configuration presets | `OpenMythos.jl/src/variants.jl` | High |
| `tests/test_main.py` | Invariant-heavy model tests | `OpenMythos.jl/test/` parity tests | Highest |
| `tests/test_tokenizer.py` | Tokenizer behavior checks | `OpenMythos.jl/test/test_tokenizer.jl` | High |
| `docs/open_mythos.md` | Architecture reference | `docs/wiki/` and future API docs | High |
| `training/3b_fine_web_edu.py` | FSDP training recipe | `OpenMythos.jl/src/training.jl` plus `OpenMythos.jl/scripts/train_3b_fineweb_edu.jl` Lux-backed bootstrap path | High |
| `docs/datasets.md` | Dataset recommendations | wiki / training docs | Medium |
| `open_mythos/moda.py` | MoDA + DeepSeek MoE branch | separate later module or deferred experiment | Low |
| `tests/bench_vs_transformer.py` | comparative benchmark | later validation tooling | Low |
| `tests/small_benchmark.py` | benchmark harness | later performance work | Low |

## Suggested Julia module breakdown

```text
OpenMythos.jl/
  src/
    OpenMythos.jl
    config.jl
    norms.jl
    rope.jl
    attention/
      gqa.jl
      mla.jl
    moe.jl
    recurrent.jl
    training.jl
    blocks.jl
    model.jl
    variants.jl
    tokenizer.jl
DeepSeekv4.jl/
  src/
    DeepSeekV4.jl
    config.jl
    attention.jl
    mhc.jl
    ffn.jl
    deepseek_model.jl
    variants.jl
```

## Practical reading notes

- `main.py` is dense but coherent: most of the port can be planned from that one file.
- `docs/open_mythos.md` is the fastest way to understand the recurrent block before reading code.
- `README.md` contains both useful implementation clues and broader theory/speculation; use it carefully.
- `moda.py` is not the same model family as the main `OpenMythos` path and should not distort the first Julia milestone.

## Commands that currently exist

These are Python-reference commands, not root-repo Julia commands:

```bash
cd reference/OpenMythos
python -m pytest tests/test_main.py -q
python -m pytest tests/test_tokenizer.py -q
python training/3b_fine_web_edu.py
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

If the Python lint tools are installed, the reference project also advertises `ruff` and `black` via `pyproject.toml`.
