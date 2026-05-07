### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 2fdb207e-a9f6-4c20-81b3-b1a3c23792d0
md"""
# OpenMythos.jl small example

This notebook builds a tiny `OpenMythos` model, runs a forward pass, and samples a short continuation.

It is intentionally self-contained and uses a bootstrap-sized config so it can run as a smoke example.
"""

# ╔═╡ 3d2ce33d-5599-4c97-8c0d-4da847bb878e
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "OpenMythos.jl"))

    using OpenMythos
    using Random
end

# ╔═╡ 90c5162d-a710-4385-b35d-6f7e7f1b58cc
md"""
## Construct a tiny model

The helper below keeps the example small and fast.
"""

# ╔═╡ 4c3e4a9a-4c33-44f6-88da-50fb01332e37
begin
    rng = MersenneTwister(1)
    cfg = bootstrap_training_config(256; seq_len=16, attn_type="gqa")
    model = OpenMythos(cfg; rng=rng)
    input_ids = reshape(collect(0:15), 1, :)
end

# ╔═╡ 7d658667-0d0c-4767-ac1b-5929084058ce
cfg

# ╔═╡ 89a8a9d7-25f2-4b4f-a67a-6f77c176ec2e
input_ids

# ╔═╡ 67c6f042-06c2-4d81-9a4f-b4ff093365b0
md"""
## Forward pass
"""

# ╔═╡ a81d6f7f-0ff8-411e-976a-c4953d81f490
logits = model(input_ids; n_loops=2)

# ╔═╡ 46e463b2-2821-4f45-b5d9-36545f7b19f0
size(logits)

# ╔═╡ 4e667062-153e-4f81-a4ab-17f1f2c264cb
begin
    next_token = argmax(vec(logits[1, end, :])) - 1
    (; next_token, max_logit=maximum(logits))
end

# ╔═╡ eaf05dc4-aa95-493a-9b16-d625a26a7102
md"""
## Generate a short continuation

The model weights are random, so the continuation is only useful as an API smoke test.
"""

# ╔═╡ be6dce37-e477-4877-8e75-b4fb2ca2d98f
generated = generate(model, input_ids[:, 1:8]; max_new_tokens=8, n_loops=2, rng=MersenneTwister(2))

# ╔═╡ 4c8bde3b-203a-42bd-b864-593480bf2b7f
generated

# ╔═╡ c2c53374-a0dc-44de-a4a9-98a4ab2bcd33
md"""
## Next steps

- replace `bootstrap_training_config` with a different config to explore shapes,
- inspect `docs/wiki/usage.md` for tokenizer and training examples,
- try `OpenMythos.jl/scripts/train_3b_fineweb_edu.jl` for the Lux-backed bootstrap training path.
"""

# ╔═╡ Cell order:
# ╟─2fdb207e-a9f6-4c20-81b3-b1a3c23792d0
# ╠═3d2ce33d-5599-4c97-8c0d-4da847bb878e
# ╟─90c5162d-a710-4385-b35d-6f7e7f1b58cc
# ╠═4c3e4a9a-4c33-44f6-88da-50fb01332e37
# ╠═7d658667-0d0c-4767-ac1b-5929084058ce
# ╠═89a8a9d7-25f2-4b4f-a67a-6f77c176ec2e
# ╟─67c6f042-06c2-4d81-9a4f-b4ff093365b0
# ╠═a81d6f7f-0ff8-411e-976a-c4953d81f490
# ╠═46e463b2-2821-4f45-b5d9-36545f7b19f0
# ╠═4e667062-153e-4f81-a4ab-17f1f2c264cb
# ╟─eaf05dc4-aa95-493a-9b16-d625a26a7102
# ╠═be6dce37-e477-4877-8e75-b4fb2ca2d98f
# ╠═4c8bde3b-203a-42bd-b864-593480bf2b7f
# ╟─c2c53374-a0dc-44de-a4a9-98a4ab2bcd33
