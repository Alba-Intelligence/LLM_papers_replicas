"""
    ManifoldHyperConnections{T}

Residual-stream mixing module inspired by manifold hyper connections.
"""
struct ManifoldHyperConnections{T<:AbstractFloat}
    n_streams::Int
    norm::RMSNorm{T}
    w_pre::Matrix{T}
    w_res::Matrix{T}
    w_post::Matrix{T}
    s_pre::Vector{T}
    s_res::Matrix{T}
    s_post::Vector{T}
    alpha_pre::T
    alpha_res::T
    alpha_post::T
    sinkhorn_iters::Int
end

function ManifoldHyperConnections(dim::Integer, n_streams::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32, sinkhorn_iters::Integer=20)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    n_streams > 0 || throw(ArgumentError("n_streams must be positive"))
    flat_dim = Int(dim * n_streams)
    return ManifoldHyperConnections(
        Int(n_streams),
        RMSNorm(flat_dim; T=T),
        T.(0.02 .* randn(rng, n_streams, flat_dim)),
        T.(0.02 .* randn(rng, n_streams * n_streams, flat_dim)),
        T.(0.02 .* randn(rng, n_streams, flat_dim)),
        zeros(T, n_streams),
        zeros(T, n_streams, n_streams),
        zeros(T, n_streams),
        T(1e-3),
        T(1e-3),
        T(1e-3),
        Int(sinkhorn_iters),
    )
end

function _sinkhorn_project(raw::AbstractMatrix{T}, niters::Integer) where {T<:AbstractFloat}
    mat = exp.(raw)
    for _ in 1:niters
        mat = mat ./ sum(mat; dims=2)
        mat = mat ./ sum(mat; dims=1)
    end
    return mat
end

function _mhc_params(mhc::ManifoldHyperConnections{T}, X::AbstractArray{T, 4}) where {T<:AbstractFloat}
    b, t, s, d = size(X)
    s == mhc.n_streams || throw(DimensionMismatch("mHC stream count mismatch"))
    params = [
        begin
            flat = vec(permutedims(X[bi, ti, :, :], (2, 1)))
            normed = vec(mhc.norm(reshape(flat, 1, :)))
            raw_a = mhc.alpha_pre .* (mhc.w_pre * normed) .+ mhc.s_pre
            raw_b = reshape(mhc.alpha_res .* (mhc.w_res * normed), s, s) .+ mhc.s_res
            raw_c = mhc.alpha_post .* (mhc.w_post * normed) .+ mhc.s_post
            (
                a=reshape(_sigmoid.(raw_a), 1, 1, s),
                b=reshape(_sinkhorn_project(raw_b, mhc.sinkhorn_iters), 1, 1, s, s),
                c=reshape(2 .* _sigmoid.(raw_c), 1, 1, s),
            )
        end
        for bi in 1:b, ti in 1:t
    ]
    A = cat([cat([params[bi, ti].a for ti in 1:t]...; dims=2) for bi in 1:b]...; dims=1)
    Bm = cat([cat([params[bi, ti].b for ti in 1:t]...; dims=2) for bi in 1:b]...; dims=1)
    C = cat([cat([params[bi, ti].c for ti in 1:t]...; dims=2) for bi in 1:b]...; dims=1)
    return A, Bm, C
end

function _mhc_collapse_streams(X::AbstractArray{T, 4}, A::AbstractArray{T, 3}) where {T<:AbstractFloat}
    b, t, s, d = size(X)
    collapsed = [
        begin
            contribs = [A[bi, ti, stream_idx] .* vec(@view X[bi, ti, stream_idx, :]) for stream_idx in 1:s]
            reshape(reduce(.+, contribs; init=zeros(T, d)), 1, 1, d)
        end
        for bi in 1:b, ti in 1:t
    ]
    return cat([cat([collapsed[bi, ti] for ti in 1:t]...; dims=2) for bi in 1:b]...; dims=1)
end

"""Collapse multi-stream state `X` into one readout stream."""
function mhc_readout(mhc::ManifoldHyperConnections{T}, X::AbstractArray{T, 4}) where {T<:AbstractFloat}
    A, _, _ = _mhc_params(mhc, X)
    return _mhc_collapse_streams(X, A)
end

"""
    mhc(X, layer_fn)

Run one mHC mixing step by collapsing `X`, applying `layer_fn`, and writing the
update back into the residual manifold.
"""
function (mhc::ManifoldHyperConnections{T})(X::AbstractArray{T, 4}, layer_fn::Function) where {T<:AbstractFloat}
    b, t, s, d = size(X)
    A, Bm, C = _mhc_params(mhc, X)
    layer_input = _mhc_collapse_streams(X, A)
    update = layer_fn(layer_input)
    next_states = [
        begin
            streams = Array(@view X[bi, ti, :, :])
            mixed = Bm[bi, ti, :, :] * streams
            reshape(mixed .+ reshape(C[bi, ti, :], s, 1) .* reshape(update[bi, ti, :], 1, d), 1, 1, s, d)
        end
        for bi in 1:b, ti in 1:t
    ]
    return cat([cat([next_states[bi, ti] for ti in 1:t]...; dims=2) for bi in 1:b]...; dims=1)
end
