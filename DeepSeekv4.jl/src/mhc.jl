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
        mat ./= sum(mat; dims=2)
        mat ./= sum(mat; dims=1)
    end
    return mat
end

function _mhc_params(mhc::ManifoldHyperConnections{T}, X::AbstractArray{T, 4}) where {T<:AbstractFloat}
    b, t, s, d = size(X)
    s == mhc.n_streams || throw(DimensionMismatch("mHC stream count mismatch"))
    A = zeros(T, b, t, s)
    Bm = zeros(T, b, t, s, s)
    C = zeros(T, b, t, s)
    for bi in 1:b, ti in 1:t
        flat = vec(permutedims(X[bi, ti, :, :], (2, 1)))
        normed = vec(mhc.norm(reshape(flat, 1, :)))
        raw_a = mhc.alpha_pre .* (mhc.w_pre * normed) .+ mhc.s_pre
        raw_b = reshape(mhc.alpha_res .* (mhc.w_res * normed), s, s) .+ mhc.s_res
        raw_c = mhc.alpha_post .* (mhc.w_post * normed) .+ mhc.s_post
        A[bi, ti, :] .= _sigmoid.(raw_a)
        Bm[bi, ti, :, :] .= _sinkhorn_project(raw_b, mhc.sinkhorn_iters)
        C[bi, ti, :] .= 2 .* _sigmoid.(raw_c)
    end
    return A, Bm, C
end

function _mhc_collapse_streams(X::AbstractArray{T, 4}, A::AbstractArray{T, 3}) where {T<:AbstractFloat}
    b, t, s, d = size(X)
    out = zeros(T, b, t, d)
    for bi in 1:b, ti in 1:t
        for stream_idx in 1:s
            @views out[bi, ti, :] .+= A[bi, ti, stream_idx] .* X[bi, ti, stream_idx, :]
        end
    end
    return out
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
    X_next = zeros(T, b, t, s, d)
    for bi in 1:b, ti in 1:t
        streams = Array(@view X[bi, ti, :, :])
        mixed = Bm[bi, ti, :, :] * streams
        @views X_next[bi, ti, :, :] .= mixed .+ reshape(C[bi, ti, :], s, 1) .* reshape(update[bi, ti, :], 1, d)
    end
    return X_next
end
