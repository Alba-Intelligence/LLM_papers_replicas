# Algorithm sketch

The implementation follows Xu & Jia, "Two-Phase Kernel Estimation for Robust Motion Deblurring" (ECCV 2010):

- Phase 1: kernel initialization with gradient confidence r (Eq. 2), mask M (Eq. 3), selective edges ∇Is (Eq. 4), FFT kernel (Eq. 6), coarse image (Eq. 8).
- Phase 2: ISD-based refinement using support detection (Eq. 9) and sparse
  regularization (Eq. 10), followed by a simple centre-of-mass
  re-centering of the kernel to avoid global circular shifts.
- Final deconvolution: TV-L1 model (Eq. 12) solved via half-quadratic
  splitting (Eqs. 13–18), using FFTs with explicit replicate padding and
  consistent kernel centering.

For full details, see the paper PDF in `docs/pdfs/`.

## Practical notes

- All heavy computation is done on a single luminance channel. This
  matches the original paper's focus on structure/edges and avoids
  colour fringing from channel-wise inconsistencies.
- FFT-based convolutions are implemented with explicit "ifftshift" of
  the padded kernel, and the estimated kernel is post-processed to
  align its centre of mass with the geometric centre. This removes the
  common quadrant-swap artefact seen in naive implementations.
- Before blind deblurring, the blurred input is passed through a simple
  cosine **edge taper** that blends a small border region towards the
  image mean. This reduces boundary discontinuities seen by the FFT and
  mitigates ringing at the outer frame.
- Despite these precautions, aggressive deconvolution of strongly
  blurred, high-frequency images can still produce moiré-like patterns
  when the kernel estimate is imperfect. The main knobs are the TV-L1
  parameters (`λ`, `max_outer`, `max_inner`) and the amount of
  smoothing/thresholding applied to the kernel in Phase 1/2. The
  default implementation chooses these TV-L1 parameters heuristically
  from the image size, but you can override them when calling
  `deblur`/`deconvolve` if needed.
