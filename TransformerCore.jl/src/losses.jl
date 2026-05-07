"""
    next_token_cross_entropy(logits, target_ids)

Return the mean next-token cross-entropy for feature-last logits of shape
`(batch, time, vocab)` and zero-based integer token targets of shape
`(batch, time)`.
"""
function next_token_cross_entropy(logits::AbstractArray{T, 3}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(logits, 1) == size(target_ids, 1) || throw(DimensionMismatch("batch size mismatch"))
    size(logits, 2) == size(target_ids, 2) || throw(DimensionMismatch("sequence length mismatch"))

    rows = permutedims(_flatten_feature_last(logits), (2, 1))
    flat_targets = vec(target_ids)
    shifted = rows .- maximum(rows; dims=2)
    log_denom = log.(sum(exp.(shifted); dims=2))
    total = zero(T)
    for i in eachindex(flat_targets)
        tgt = flat_targets[i] + 1
        (1 <= tgt <= size(rows, 2)) || throw(BoundsError(rows, (i, tgt)))
        total -= shifted[i, tgt] - log_denom[i]
    end
    return total / T(length(flat_targets))
end
