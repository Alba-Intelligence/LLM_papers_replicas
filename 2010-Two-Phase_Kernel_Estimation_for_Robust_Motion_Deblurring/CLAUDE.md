# CLAUDE.md

This file exists for agents that look for `CLAUDE.md`.

The repository-wide source of truth is `AGENTS.md`.

For any work under `docs/private/llm_knowledge/`, also follow
`docs/private/llm_knowledge/AGENTS.md`.

Short project snapshot:

- Julia replication of Xu & Jia (ECCV 2010) two-phase kernel estimation for
  robust motion deblurring
- Two main packages: `TVL1Deconv` for algorithm internals and
  `RobustMotionDeblur` for the public API
- Heavy computation currently runs in luminance space, so deblurred outputs are
  grayscale unless the implementation is explicitly extended

Critical reminder: when kernel alignment or FFT wrap-around artefacts appear,
inspect the Phase 1/Phase 2 centering helpers and the TV-L1 solver padding and
FFT-shift logic before changing unrelated code.
