mutable struct KVCacheEnvelope
    cache::Dict{String, Any}
    start_pos::Int
    capacity_hint::Int

    function KVCacheEnvelope(cache::Dict{String, Any}, start_pos::Integer, capacity_hint::Integer=0)
        start_pos >= 0 || throw(ArgumentError("start_pos must be non-negative"))
        capacity_hint >= 0 || throw(ArgumentError("capacity_hint must be non-negative"))
        new(cache, Int(start_pos), Int(capacity_hint))
    end
end

KVCacheEnvelope() = KVCacheEnvelope(Dict{String, Any}(), 0)
KVCacheEnvelope(; capacity_hint::Integer=0) = KVCacheEnvelope(Dict{String, Any}(), 0, capacity_hint)

function reserve_kv_capacity!(envelope::KVCacheEnvelope, required_tokens::Integer)
    required_tokens >= 0 || throw(ArgumentError("required_tokens must be non-negative"))
    envelope.capacity_hint = max(envelope.capacity_hint, Int(required_tokens))
    return envelope
end

function save_kv_cache(envelope::KVCacheEnvelope, path::AbstractString)
    mkpath(dirname(path))
    temp_path = path * ".tmp"
    payload = Dict(
        "cache" => envelope.cache,
        "start_pos" => envelope.start_pos,
        "capacity_hint" => envelope.capacity_hint,
    )
    open(temp_path, "w") do io
        serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

function load_kv_cache(path::AbstractString)
    payload = open(path, "r") do io
        deserialize(io)
    end
    cache = payload["cache"]
    cache isa Dict{String, Any} || throw(ArgumentError("serialized cache payload is not a Dict{String, Any}"))
    return KVCacheEnvelope(cache, Int(payload["start_pos"]), Int(get(payload, "capacity_hint", 0)))
end
