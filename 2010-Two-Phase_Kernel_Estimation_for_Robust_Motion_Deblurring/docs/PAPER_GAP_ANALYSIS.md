# Gap analysis: implementation vs Xu & Jia (ECCV 2010)

This document compares the current codebase to the paper **"Two-Phase Kernel Estimation for Robust Motion Deblurring"** (Li Xu, Jiaya Jia, ECCV 2010). The paper was read from the project PDF (same content as on the [project page](http://www.cse.cuhk.edu.hk/~leojia/projects/robust_deblur/)).

---

## What the paper emphasizes

1. **Edge selection**: Strong edges do *not* always help; narrow structures (smaller than the kernel) can degrade estimation. The **r map** (Eq. 2) and **mask M** (Eq. 3) select useful edges; **∇I_s** (Eq. 4) is the selective edge map.
2. **Phase 1**: Multi-scale kernel init with Gaussian prior (Eq. 6), **spatial prior** for coarse image (Eq. 7–8: ‖∇I − ∇I_s‖²), and decaying τr, τs.
3. **Phase 2**: **ISD avoids hard thresholding**. Support S is detected (Eq. 9); then the energy **Eq. (10)** is minimized with L1 only on the *complement* of S, solved by **IRLS** (Eq. 11). Result: “soft-threshold-like” sparsity without zeroing out small elements by hand.
4. **TV-L1 deconvolution**: L1 data term (Eq. 12), half-quadratic with auxiliary variables v and w (Eq. 13–18), Algorithm 3. Edge taping, warm-start β/θ.
5. **Parameters**: λ = 2e−3, γ = 10 in Phase 1; γ = 1 in Phase 2; λ = 2e−2 in deconvolution. τr, τs set adaptively as in [5] (4 gradient directions; τs so ≥ 2√Pk pixels per group; τr so ≥ 0.5√(PI·Pk) per group).

---

## Implemented and aligned

| Paper element | Status |
|---------------|--------|
| Gradient confidence r (Eq. 2) | ✓ `gradient_confidence` in `edges.jl` |
| Mask M = H(r − τr) (Eq. 3) | ✓ in `select_edges` |
| Selective edges ∇I_s (Eq. 4) | ✓ in `select_edges` |
| Kernel from Gaussian prior (Eq. 6) | ✓ `fft_kernel_from_gradients` in `phase1.jl` |
| Coarse image with spatial prior (Eq. 7–8) | ✓ `coarse_image_from_kernel` uses ‖∇I − ∇I_s‖² and FFT closed form (Eq. 8) |
| Multi-scale pyramid, τr/τs decay | ✓ Phase 1 loop; decay_factor (e.g. 1.1) |
| First significant jump for support (Eq. 9) | ✓ `first_significant_jump` in `phase2.jl` |
| TV-L1 model (Eq. 12), half-quadratic (13–18), Alg. 3 | ✓ `solver.jl` (v, w, Eq. 15/17/18) |
| Edge taping | ✓ `edgetaper` in RobustMotionDeblur |
| Default λ, γ in the right ballpark | ✓ |

---

## Gaps (missing or different from the paper)

### 1. Phase 2: real ISD minimization (Eq. 10–11) — **major**

- **Paper**: Iteratively (a) set support S = {j : k_j > s} with “first significant jump” for s; (b) **minimize**  
  **E(k) = (1/2)‖∇I_s ⊗ k − ∇B‖² + γ Σ_{j ∈ S̄} |k_j|**  
  (L1 only on the *complement* S̄ of S). Solve via **IRLS** (Eq. 11):  
  [A^T A + γ diag(V_S̄ Ψ^{−1})] V_k = A^T V_B, with Ψ = max(|V_k^{t−1}|, 1e−5). So large-value elements are *less penalized*; sparsity is achieved by adaptive regularization, not by zeroing.
- **Current code**: Uses support only to **hard-threshold** (zero out k below s), then renormalize (and optional box smooth, min_frac_max). **No minimization of Eq. (10)** and **no IRLS**. So we do the opposite of the paper’s “avoids hard threshold of the kernel elements.”

**Recommendation**: Implement Phase 2 as in Algorithm 2: each iteration solve Eq. (10) with IRLS (Eq. 11) for the current support, then update S from the new k. This would replace (or sit behind) the current hard-threshold + renormalize loop.

---

### 2. Shock filter: second derivative (Eq. 1)

- **Paper**: ∂I/∂t = −sign(ΔI)‖∇I‖ with  
  **ΔI = I_x² I_xx + 2 I_x I_y I_xy + I_y² I_yy**  
  (second derivative in the gradient direction, not Laplacian).
- **Current code**: `edges.jl` uses **ΔI = laplacian(I)** (I_xx + I_yy).

So we use a different ΔI; edge sharpening may differ from the paper.

**Recommendation**: Implement the directional second derivative for the shock filter (Eq. 1) and use it in `shock_filter`.

---

### 3. Adaptive τr, τs (Section 4, “[5]”)

- **Paper**: “We adaptively set the initial values of τr and τs, using the method of [5]. The directions of image gradient are initially quantized into four groups. τs is set to guarantee that at least 2√Pk pixels participate in kernel estimation in each group…. τr is similarly determined by allowing at least 0.5√(PI·Pk) pixels to be selected in each group.”
- **Current code**: Fixed `τr_init`, `τs_init` (e.g. 0.2, 1e−4) and decay; no gradient-direction quantization or Pk/PI-based rules.

**Recommendation**: Add an optional adaptive initialization for τr and τs following the paper’s description (4 gradient bins, 2√Pk and 0.5√(PI·Pk) rules).

---

### 4. Pyramid: upscaling latent to next level (Alg. 1)

- **Paper**: “Upscale image I_{l+1} ← I_l ↑” — after each pyramid level, the latent image I is **upsampled** to initialize the next (finer) level.
- **Current code**: Pyramid is built by downsampling B; each level runs on downsampled B. The latent I is not explicitly upsampled and passed as the starting image for the next level.

So we may be missing the “warm start” of the latent from coarser to finer scale (we do blend latent with blurred within a level for edge selection).

**Recommendation**: After each level, upsample the current latent I (e.g. by 2×) and use it as the initial image (or for edge selection) at the next finer level, as in Algorithm 1.

---

## Summary

- **Phase 1** is largely aligned (r, M, ∇I_s, Eq. 6, Eq. 8, multi-scale, decay). Gaps: shock filter uses Laplacian instead of directional second derivative; τr/τs are not adaptively set as in the paper; no explicit upscale of I between pyramid levels.
- **Phase 2** is the largest gap: the paper solves **Eq. (10)** with IRLS (Eq. 11) and adaptive support S, and explicitly avoids hard thresholding; our Phase 2 is support detection + hard threshold + renormalize, with no minimization step.
- **TV-L1** and high-level pipeline match the paper well.

Implementing **Phase 2 as in the paper (Eq. 10 + IRLS)** and, if desired, the **shock filter (Eq. 1)** and **adaptive τr/τs** would bring the implementation closest to the original method.
