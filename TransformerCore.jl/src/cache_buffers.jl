mutable struct AxisAppendBuffer{T, N}
    data::Array{T, N}
    length::Int
    axis::Int

    function AxisAppendBuffer(data::Array{T, N}, length::Integer, axis::Integer) where {T, N}
        1 <= axis <= N || throw(ArgumentError("axis must be between 1 and $N"))
        0 <= length <= size(data, axis) || throw(ArgumentError("length must fit within buffer capacity"))
        new{T, N}(data, Int(length), Int(axis))
    end
end

function AxisAppendBuffer(chunk::AbstractArray{T, N}, axis::Integer=2; capacity::Union{Nothing, Integer}=nothing) where {T, N}
    1 <= axis <= N || throw(ArgumentError("axis must be between 1 and $N"))
    chunk_len = size(chunk, axis)
    cap = capacity === nothing ? chunk_len : Int(capacity)
    cap >= chunk_len || throw(ArgumentError("capacity must be at least the chunk length"))
    dims = collect(size(chunk))
    dims[Int(axis)] = cap
    return AxisAppendBuffer(Array{T}(undef, Tuple(dims)), 0, axis)
end

function filled_axis_buffer(chunk::AbstractArray{T, N}; axis::Integer=2) where {T, N}
    buffer = AxisAppendBuffer(chunk, axis)
    append_axis_buffer!(buffer, chunk)
    return buffer
end

function _buffer_ranges(buffer::AxisAppendBuffer, axis_range)
    return ntuple(i -> i == buffer.axis ? axis_range : Colon(), ndims(buffer.data))
end

function _check_axis_compatible(buffer::AxisAppendBuffer{T, N}, chunk::AbstractArray{S, N}) where {T, S, N}
    eltype(chunk) == T || throw(ArgumentError("chunk element type $(eltype(chunk)) does not match buffer element type $T"))
    for dim_idx in 1:N
        dim_idx == buffer.axis && continue
        size(buffer.data, dim_idx) == size(chunk, dim_idx) || throw(DimensionMismatch("chunk shape does not match buffer shape outside the append axis"))
    end
    return nothing
end

function ensure_axis_capacity!(buffer::AxisAppendBuffer, required_length::Integer)
    required = Int(required_length)
    required >= 0 || throw(ArgumentError("required_length must be non-negative"))
    current_capacity = size(buffer.data, buffer.axis)
    required <= current_capacity && return buffer

    new_capacity = max(required, max(1, current_capacity * 2))
    dims = collect(size(buffer.data))
    dims[buffer.axis] = new_capacity
    new_data = Array{eltype(buffer.data)}(undef, Tuple(dims))
    if buffer.length > 0
        ranges = _buffer_ranges(buffer, 1:buffer.length)
        @views new_data[ranges...] .= buffer.data[ranges...]
    end
    buffer.data = new_data
    return buffer
end

function append_axis_buffer!(buffer::AxisAppendBuffer{T, N}, chunk::AbstractArray{T, N}) where {T, N}
    _check_axis_compatible(buffer, chunk)
    chunk_len = size(chunk, buffer.axis)
    ensure_axis_capacity!(buffer, buffer.length + chunk_len)
    dest_range = (buffer.length + 1):(buffer.length + chunk_len)
    ranges = _buffer_ranges(buffer, dest_range)
    @views buffer.data[ranges...] .= chunk
    buffer.length += chunk_len
    return buffer
end

function buffer_view(buffer::AxisAppendBuffer)
    ranges = _buffer_ranges(buffer, 1:buffer.length)
    return @view buffer.data[ranges...]
end

Base.:(==)(a::AxisAppendBuffer, b::AxisAppendBuffer) =
    a.axis == b.axis && a.length == b.length && Array(buffer_view(a)) == Array(buffer_view(b))
