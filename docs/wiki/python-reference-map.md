# OpenMythos Python Reference Map

This page is the bridge between the upstream Python OpenMythos project and the Julia workspace.

It is intentionally about the **OpenMythos** reference tree in `reference/private/OpenMythos`.

DeepSeek V4 source mapping lives in [DeepSeek V4 reference map](deepseek-v4-reference-map.md).

## Source-of-truth order

1. `reference/private/OpenMythos/open_mythos/main.py` - ground-truth behavior.
2. `reference/private/OpenMythos/docs/open_mythos.md` - best prose description of the architecture.
3. `reference/private/OpenMythos/tests/test_main.py` - executable invariants.
4. `reference/private/OpenMythos/open_mythos/variants.py` - preset model scales.
5. `reference/private/OpenMythos/training/3b_fine_web_edu.py` - current training workflow.
6. `reference/private/OpenMythos/README.md` - useful overview, but more speculative and less implementation-tight.

## File-by-file mapping

| Python path                     | Purpose                                        | Julia target                                                                                                                                                                                    | Priority |
| ------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `open_mythos/main.py`           | Core model, layers, recurrent loop, generation | `OpenMythos.jl/src/` core modules and model assembly                                                                                                                                            | Highest  |
| `open_mythos/tokenizer.py`      | Hugging Face tokenizer wrapper                 | `OpenMythos.jl/src/tokenizer.jl` via the current Julia-native BytePairEncoding / tiktoken-style wrapper                                                                                         | High     |
| `open_mythos/variants.py`       | Named configuration presets                    | `OpenMythos.jl/src/variants.jl`                                                                                                                                                                 | High     |
| `tests/test_main.py`            | Invariant-heavy model tests                    | `OpenMythos.jl/test/` parity tests                                                                                                                                                              | Highest  |
| `tests/test_tokenizer.py`       | Tokenizer behavior checks                      | `OpenMythos.jl/test/test_tokenizer.jl`                                                                                                                                                          | High     |
| `docs/open_mythos.md`           | Architecture reference                         | `docs/wiki/architecture.md` and package docs                                                                                                                                                    | High     |
| `training/3b_fine_web_edu.py`   | FSDP training recipe                           | `OpenMythos.jl/src/training.jl` plus `OpenMythos.jl/scripts/train_3b_fineweb_edu.jl` Lux-backed bootstrap path with Julia-native tokenizer, local parquet, and remote rows-api smoke data paths | High     |
| `docs/datasets.md`              | Dataset recommendations                        | wiki / training docs                                                                                                                                                                            | Medium   |
| `open_mythos/moda.py`           | MoDA + DeepSeek MoE branch                     | separate later module or deferred experiment                                                                                                                                                    | Low      |
| `tests/bench_vs_transformer.py` | comparative benchmark                          | later validation tooling                                                                                                                                                                        | Low      |
| `tests/small_benchmark.py`      | benchmark harness                              | later performance work                                                                                                                                                                          | Low      |

## Suggested Julia module breakdown

```text
OpenMythos.jl/
  src/
    OpenMythos.jl
    config.jl
    attention.jl
    ffn.jl
    blocks.jl
    recurrent.jl
    model.jl
    training.jl
    variants.jl
    tokenizer.jl
TransformerCore.jl/
  src/
    TransformerCore.jl
    norms.jl
    rope.jl
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

- `main.py` is dense but coherent: most of the OpenMythos port can be planned from that one file.
- `docs/open_mythos.md` is the fastest way to understand the recurrent block before reading code.
- `README.md` contains both useful implementation clues and broader theory/speculation; use it carefully.
- `moda.py` is not the same model family as the main `OpenMythos` path and should not distort the first Julia milestone.
- The Julia workspace now separates shared generic code into `TransformerCore.jl/`, so not every Python concept maps directly into `OpenMythos.jl/src/`.

## Commands that currently exist

These are Python-reference commands, not root-repo Julia commands:

```bash
cd reference/private/OpenMythos
python -m pytest tests/test_main.py -q
python -m pytest tests/test_tokenizer.py -q
python training/3b_fine_web_edu.py
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

If the Python lint tools are installed, the reference project also advertises `ruff` and `black` via `pyproject.toml`.
