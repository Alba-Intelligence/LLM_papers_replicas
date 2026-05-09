# References

This page consolidates the references currently shaping the Julia workspace:

- the PDF/article/YAML materials driving the broader architecture-comparison program,
- citations inherited from the Python OpenMythos project,
- DeepSeek V4 materials used for the new package,
- dataset and implementation references that inform future training/runtime work.

## Architecture comparison materials

| Topic                                  | Reference                                                | URL / Path                                                                  | Used for                                                                                          |
| -------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| PDF snapshot of the comparison article | Sebastian Raschka, _The Big LLM Architecture Comparison_ | `reference/private/The Big LLM Architecture Comparison.pdf`                 | workspace-level architecture inventory and future family scoping                                  |
| Live article version                   | Sebastian Raschka, _The Big LLM Architecture Comparison_ | https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison | cross-checking section order, updates, and explanatory text                                       |
| Structured model registry              | `Big-LLM-Architecture-models.yml`                        | `reference/private/Big-LLM-Architecture-models.yml`                         | authoritative model names, config links, tech-report links, and metadata for PDF-covered families |
| Gemma 3n authority for the PDF scope   | Google Developers Blog, _Introducing Gemma 3n_           | https://developers.googleblog.com/en/introducing-gemma-3n/                  | authority for the Gemma 3n section referenced by the comparison map                               |

## DeepSeek V4 materials

| Topic                      | Reference                  | URL                                                                          | Used for                                                     |
| -------------------------- | -------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------ |
| DeepSeek V4 technical note | DeepSeek V4 PDF            | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf | `DeepSeekv4.jl/`, DeepSeek wiki pages, architecture planning |
| DeepSeek V4 introduction   | YouTube introduction video | https://www.youtube.com/watch?v=q8holiIirgo                                  | architecture planning and terminology cross-checking         |

## DeepSeek V4 lineage and secondary sources

These references help explain the late-2025 / early-2026 research trajectory around DeepSeek V4, but they are **secondary** to the official V4 technical note. Use them for context, tradeoff analysis, and future-planning ideas rather than as binding implementation spec.

| Topic                                | Reference                                 | URL                                                                                                          | Used for                                                                          |
| ------------------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Pre-release V4 analysis              | Kili Technology DeepSeek V4 article       | https://kili-technology.com/blog/data-story-deepseek-v4                                                      | DeepSeek roadmap context, data/training interpretation, speculative lineage notes |
| Conditional memory                   | Engram paper                              | https://arxiv.org/abs/2601.07372                                                                             | future DeepSeek research branch, memory/compute separation context                |
| Conditional memory implementation    | Engram official repo                      | https://github.com/deepseek-ai/Engram                                                                        | architecture sketches and demo code for conditional-memory experiments            |
| Residual stability at scale          | mHC paper                                 | https://arxiv.org/abs/2512.24880                                                                             | residual-mixing lineage and scaling-stability context                             |
| DeepSeek training lineage            | DeepSeek-V3 technical report              | https://arxiv.org/html/2412.19437v1                                                                          | MLA/MoE/MTP lineage, data construction, SFT/RL and systems context                |
| Sparse-attention release speculation | Tom's Hardware on Engram / memory offload | https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-touts-memory-breakthrough-engram | secondary commentary on host-memory/offloading implications                       |

## Architecture and reasoning

| Topic                       | Reference                                                                       | URL                              | Referenced in                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Recurrent-Depth Transformer | Loop, Think, & Generalize                                                       | https://arxiv.org/pdf/2604.07822 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`                                                     |
| Stable looped training      | Parcae / Scaling Laws for Stable Looped Language Models                         | https://arxiv.org/abs/2604.12946 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py` |
| Looped reasoning            | Reasoning with Latent Thoughts                                                  | https://arxiv.org/abs/2502.17416 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py` |
| ACT basis                   | Universal Transformers                                                          | https://arxiv.org/pdf/1807.03819 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`                                                     |
| Continuous latent reasoning | COCONUT / Training Large Language Models to Reason in a Continuous Latent Space | https://arxiv.org/abs/2412.06769 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`                                                     |
| Recurrent alternatives      | Hyperloop Transformers                                                          | https://arxiv.org/abs/2604.21254 | `README.md`                                                                                                         |
| Recurrent alternatives      | The Recurrent Transformer: Greater Effective Depth and Efficient Decoding       | https://arxiv.org/abs/2604.21215 | `README.md`                                                                                                         |

## Attention, MoE, and core building blocks

| Topic                                     | Reference                      | URL                              | Referenced in                                                                                                                                                           |
| ----------------------------------------- | ------------------------------ | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepSeek V4 predecessor attention lineage | DeepSeek-V2                    | https://arxiv.org/abs/2405.04434 | `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`, DeepSeek V4 implementation context                              |
| MLA                                       | DeepSeek-V2                    | https://arxiv.org/abs/2405.04434 | `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`                                                                  |
| GQA                                       | Ainslie et al., 2023           | https://arxiv.org/abs/2305.13245 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`                                                     |
| Flash Attention 2                         | Dao et al., 2023               | cited by name in repo            | `README.md`, `reference/private/OpenMythos/open_mythos/main.py`                                                                                                         |
| MoE FFN                                   | DeepSeekMoE                    | https://arxiv.org/abs/2401.06066 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`, `reference/private/OpenMythos/open_mythos/moda.py` |
| Depth-wise LoRA                           | Relaxed Recursive Transformers | https://arxiv.org/pdf/2410.20672 | `README.md`, `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`                                                     |
| RMSNorm                                   | Zhang and Sennrich, 2019       | https://arxiv.org/abs/1910.07467 | `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`                                                                  |
| RoPE                                      | Su et al., 2021                | https://arxiv.org/abs/2104.09864 | `reference/private/OpenMythos/docs/open_mythos.md`                                                                                                                      |
| ACT                                       | Graves, 2016                   | https://arxiv.org/abs/1603.08983 | `reference/private/OpenMythos/docs/open_mythos.md`, `reference/private/OpenMythos/open_mythos/main.py`                                                                  |

## MoDA branch references

| Topic                                | Reference                   | URL                                        | Referenced in                                                   |
| ------------------------------------ | --------------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| MoDA attention                       | Mixture-of-Depths Attention | https://arxiv.org/abs/2603.15619           | `README.md`, `reference/private/OpenMythos/open_mythos/moda.py` |
| DeepSeek V3 implementation reference | DeepSeek-V3 repo            | https://github.com/deepseek-ai/DeepSeek-V3 | `reference/private/OpenMythos/open_mythos/moda.py`              |
| MoDA kernel reference                | hustvl/MoDA                 | https://github.com/hustvl/MoDA             | `reference/private/OpenMythos/open_mythos/moda.py`              |

## Datasets

| Dataset        | URL                                                         | Referenced in                                                                                               |
| -------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| FineWeb-Edu    | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu   | `reference/private/OpenMythos/docs/datasets.md`, `reference/private/OpenMythos/training/3b_fine_web_edu.py` |
| OpenHermes 2.5 | https://huggingface.co/datasets/teknium/OpenHermes-2.5      | `reference/private/OpenMythos/docs/datasets.md`                                                             |
| OpenWebMath    | https://huggingface.co/datasets/open-web-math/open-web-math | `reference/private/OpenMythos/docs/datasets.md`                                                             |

## Project and implementation references

| Reference                   | URL                                                 | Referenced in                                   |
| --------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| Python reference repository | https://github.com/The-Swarm-Corporation/OpenMythos | `pyproject.toml`, citation block in `README.md` |
| PyPI package                | https://pypi.org/project/open-mythos/               | `README.md`                                     |
| DeepSeek V4 model page      | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro  | DeepSeek package planning                       |
| DeepSeek API/news summary   | https://api-docs.deepseek.com/news/news250929       | DeepSeek release context                        |

## Discussion threads cited in the Python README

| Topic                                                 | URL                                                     |
| ----------------------------------------------------- | ------------------------------------------------------- |
| Sigrid Jin on looped transformer theory               | https://x.com/realsigridjin/status/2044620031410266276  |
| Yuekun Yao on implicit reasoning                      | https://x.com/yuekun_yao/status/2044229171627639004     |
| rosinality on cyclic trajectories and input injection | https://x.com/rosinality/status/2043953033428541853     |
| Hayden Prairie on Parcae scaling laws                 | https://x.com/hayden_prairie/status/2044453231913537927 |
| davidad on loop index embedding                       | https://x.com/davidad/status/2044453231913537927        |
| Chris Hayduk on looped transformer controversy        | https://x.com/ChrisHayduk/status/2045947623572688943    |
| Sigrid Jin summary thread                             | https://x.com/realsigridjin/status/2046012743778766875  |
