mutable struct KVCacheEnvelope
    cache::Dict{String, Any}
    start_pos::Int

    function KVCacheEnvelope(cache::Dict{String, Any}, start_pos::Integer)
        start_pos >= 0 || throw(ArgumentError("start_pos must be non-negative"))
        new(cache, Int(start_pos))
    end
end

KVCacheEnvelope() = KVCacheEnvelope(Dict{String, Any}(), 0)

function save_kv_cache(envelope::KVCacheEnvelope, path::AbstractString)
    mkpath(dirname(path))
    temp_path = path * ".tmp"
    payload = Dict(
        "cache" => envelope.cache,
        "start_pos" => envelope.start_pos,
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
    return KVCacheEnvelope(cache, Int(payload["start_pos"]))
end
