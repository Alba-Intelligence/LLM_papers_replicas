import sys

import torch

sys.path.insert(0, "../reference/OpenMythos")

from open_mythos.main import apply_rope, precompute_rope_freqs


def deterministic_x(b: int, t: int, h: int, d: int) -> torch.Tensor:
    total = b * t * h * d
    vals = torch.arange(1, total + 1, dtype=torch.float32) / 100.0
    return vals.view(b, t, h, d)


def print_matrix(mat: torch.Tensor) -> None:
    print(",".join(str(float(x)) for x in mat.reshape(-1).tolist()))


def main() -> None:
    action = sys.argv[1]
    if action == "precompute_rope":
        dim = int(sys.argv[2])
        max_len = int(sys.argv[3])
        theta = float(sys.argv[4])
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len, theta=theta)
        print(freqs.shape[0], freqs.shape[1])
        print_matrix(freqs.real)
        print_matrix(freqs.imag)
        return

    if action == "apply_rope":
        b = int(sys.argv[2])
        t = int(sys.argv[3])
        h = int(sys.argv[4])
        d = int(sys.argv[5])
        theta = float(sys.argv[6])
        x = deterministic_x(b, t, h, d)
        freqs = precompute_rope_freqs(dim=d, max_len=t, theta=theta)
        out = apply_rope(x, freqs)
        print(b, t, h, d)
        print_matrix(out)
        return

    raise ValueError(f"unknown action: {action}")


if __name__ == "__main__":
    main()
