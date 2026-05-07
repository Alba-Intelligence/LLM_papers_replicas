const _CHECKPOINT_KIND = "transformer_core.next_token_trainer"

_checkpoint_name(step::Integer) = "step_$(lpad(string(step), 7, '0')).jls"

function _checkpoint_files(dir::AbstractString)
    isdir(dir) || return String[]
    files = filter(name -> startswith(name, "step_") && endswith(name, ".jls"), readdir(dir))
    sort!(files)
    return files
end

"""
    checkpoint_dir(root; family, mode)

Return the canonical checkpoint directory for a model `family` and training
`mode` under `root`.
"""
function checkpoint_dir(root::AbstractString; family::AbstractString, mode::AbstractString)
    isempty(family) && throw(ArgumentError("family must be non-empty"))
    isempty(mode) && throw(ArgumentError("mode must be non-empty"))
    return joinpath(root, family, mode)
end

"""
    latest_checkpoint(root; family=nothing, mode=nothing)

Return the newest checkpoint path. Without `family`/`mode`, preserves the legacy
behavior of scanning `root` directly.
"""
function latest_checkpoint(root::AbstractString; family::Union{Nothing, AbstractString}=nothing, mode::Union{Nothing, AbstractString}=nothing)
    if (family === nothing) != (mode === nothing)
        throw(ArgumentError("pass both family and mode, or neither"))
    end
    dir = family === nothing ? root : checkpoint_dir(root; family=family, mode=mode)
    files = _checkpoint_files(dir)
    isempty(files) && return nothing
    return joinpath(dir, last(files))
end

function _prune_checkpoints!(dir::AbstractString, keep_last::Integer)
    keep_last > 0 || throw(ArgumentError("keep_last must be positive"))
    files = _checkpoint_files(dir)
    for old in files[1:max(0, length(files) - keep_last)]
        rm(joinpath(dir, old); force=true)
    end
    return nothing
end

"""
    save_trainer_checkpoint(state, root; family, mode, keep_last=3, config=nothing, metadata=Dict())

Serialize a `NextTokenTrainerState` into a family/mode-specific checkpoint
subdirectory.
"""
function save_trainer_checkpoint(
    state::NextTokenTrainerState,
    root::AbstractString;
    family::AbstractString,
    mode::AbstractString,
    keep_last::Integer=3,
    config=nothing,
    metadata::AbstractDict=Dict{String, Any}(),
)
    dir = checkpoint_dir(root; family=family, mode=mode)
    mkpath(dir)
    final_path = joinpath(dir, _checkpoint_name(state.step))
    temp_path = final_path * ".tmp"
    payload = Dict(
        "kind" => _CHECKPOINT_KIND,
        "family" => String(family),
        "mode" => String(mode),
        "step" => state.step,
        "parameters" => state.ps,
        "states" => state.st,
        "opt_state" => state.opt_state,
        "schedule" => state.schedule,
        "config" => config,
        "metadata" => Dict(string(k) => v for (k, v) in pairs(metadata)),
    )
    open(temp_path, "w") do io
        serialize(io, payload)
    end
    mv(temp_path, final_path; force=true)
    _prune_checkpoints!(dir, keep_last)
    return final_path
end

"""
    load_trainer_checkpoint(path; expected_family=nothing, expected_mode=nothing)

Load a generic next-token trainer checkpoint and return a named tuple with the
restored `state` plus payload metadata.
"""
function load_trainer_checkpoint(
    path::AbstractString;
    expected_family::Union{Nothing, AbstractString}=nothing,
    expected_mode::Union{Nothing, AbstractString}=nothing,
)
    payload = open(path, "r") do io
        deserialize(io)
    end
    get(payload, "kind", nothing) == _CHECKPOINT_KIND || throw(ArgumentError("checkpoint is not a TransformerCore next-token trainer checkpoint"))
    expected_family === nothing || payload["family"] == expected_family || throw(ArgumentError("checkpoint family mismatch"))
    expected_mode === nothing || payload["mode"] == expected_mode || throw(ArgumentError("checkpoint mode mismatch"))

    state = NextTokenTrainerState(
        payload["parameters"],
        payload["states"],
        payload["opt_state"],
        payload["schedule"],
        Int(payload["step"]),
    )
    return (
        state=state,
        family=payload["family"],
        mode=payload["mode"],
        config=get(payload, "config", nothing),
        metadata=get(payload, "metadata", Dict{String, Any}()),
    )
end
