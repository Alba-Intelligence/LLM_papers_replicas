# References Pulled from the Python Codebase

This page consolidates references already cited in the Python reference project so the Julia port can reuse them consistently.

## Architecture and reasoning

| Topic | Reference | URL | Referenced in |
| --- | --- | --- | --- |
| Recurrent-Depth Transformer | Loop, Think, & Generalize | https://arxiv.org/pdf/2604.07822 | `README.md`, `docs/open_mythos.md` |
| Stable looped training | Parcae / Scaling Laws for Stable Looped Language Models | https://arxiv.org/abs/2604.12946 | `README.md`, `docs/open_mythos.md`, `open_mythos/main.py` |
| Looped reasoning | Reasoning with Latent Thoughts | https://arxiv.org/abs/2502.17416 | `README.md`, `docs/open_mythos.md`, `open_mythos/main.py` |
| ACT basis | Universal Transformers | https://arxiv.org/pdf/1807.03819 | `README.md`, `docs/open_mythos.md` |
| Continuous latent reasoning | COCONUT / Training Large Language Models to Reason in a Continuous Latent Space | https://arxiv.org/abs/2412.06769 | `README.md`, `docs/open_mythos.md` |
| Recurrent alternatives | Hyperloop Transformers | https://arxiv.org/abs/2604.21254 | `README.md` |
| Recurrent alternatives | The Recurrent Transformer: Greater Effective Depth and Efficient Decoding | https://arxiv.org/abs/2604.21215 | `README.md` |

## Attention, MoE, and core building blocks

| Topic | Reference | URL | Referenced in |
| --- | --- | --- | --- |
| MLA | DeepSeek-V2 | https://arxiv.org/abs/2405.04434 | `docs/open_mythos.md`, `open_mythos/main.py` |
| GQA | Ainslie et al., 2023 | https://arxiv.org/abs/2305.13245 | `README.md`, `docs/open_mythos.md`, `open_mythos/main.py` |
| Flash Attention 2 | Dao et al., 2023 | cited by name in repo | `README.md`, `open_mythos/main.py` |
| MoE FFN | DeepSeekMoE | https://arxiv.org/abs/2401.06066 | `README.md`, `docs/open_mythos.md`, `open_mythos/main.py`, `open_mythos/moda.py` |
| Depth-wise LoRA | Relaxed Recursive Transformers | https://arxiv.org/pdf/2410.20672 | `README.md`, `docs/open_mythos.md`, `open_mythos/main.py` |
| RMSNorm | Zhang and Sennrich, 2019 | https://arxiv.org/abs/1910.07467 | `docs/open_mythos.md`, `open_mythos/main.py` |
| RoPE | Su et al., 2021 | https://arxiv.org/abs/2104.09864 | `docs/open_mythos.md` |
| ACT | Graves, 2016 | https://arxiv.org/abs/1603.08983 | `docs/open_mythos.md`, `open_mythos/main.py` |

## MoDA branch references

| Topic | Reference | URL | Referenced in |
| --- | --- | --- | --- |
| MoDA attention | Mixture-of-Depths Attention | https://arxiv.org/abs/2603.15619 | `README.md`, `open_mythos/moda.py` |
| DeepSeek V3 implementation reference | DeepSeek-V3 repo | https://github.com/deepseek-ai/DeepSeek-V3 | `open_mythos/moda.py` |
| MoDA kernel reference | hustvl/MoDA | https://github.com/hustvl/MoDA | `open_mythos/moda.py` |

## Datasets

| Dataset | URL | Referenced in |
| --- | --- | --- |
| FineWeb-Edu | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | `docs/datasets.md`, `training/3b_fine_web_edu.py` |
| OpenHermes 2.5 | https://huggingface.co/datasets/teknium/OpenHermes-2.5 | `docs/datasets.md` |
| OpenWebMath | https://huggingface.co/datasets/open-web-math/open-web-math | `docs/datasets.md` |

## Project and implementation references

| Reference | URL | Referenced in |
| --- | --- | --- |
| Python reference repository | https://github.com/The-Swarm-Corporation/OpenMythos | `pyproject.toml`, citation block in `README.md` |
| PyPI package | https://pypi.org/project/open-mythos/ | `README.md` |

## Discussion threads cited in the Python README

| Topic | URL |
| --- | --- |
| Sigrid Jin on looped transformer theory | https://x.com/realsigridjin/status/2044620031410266276 |
| Yuekun Yao on implicit reasoning | https://x.com/yuekun_yao/status/2044229171627639004 |
| rosinality on cyclic trajectories and input injection | https://x.com/rosinality/status/2043953033428541853 |
| Hayden Prairie on Parcae scaling laws | https://x.com/hayden_prairie/status/2044453231913537927 |
| davidad on loop index embedding | https://x.com/davidad/status/2044453231913537927 |
| Chris Hayduk on looped transformer controversy | https://x.com/ChrisHayduk/status/2045947623572688943 |
| Sigrid Jin summary thread | https://x.com/realsigridjin/status/2046012743778766875 |

