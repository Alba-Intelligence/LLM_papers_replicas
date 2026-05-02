abstract type AbstractAxisBuffer{T, N} end

"""
    AxisAppendBuffer{T,N}

Contiguous single-allocation append buffer used for backward-compatible cache
payloads.
"""
mutable struct AxisAppendBuffer{T, N} <: AbstractAxisBuffer{T, N}
    data::Array{T, N}
    length::Int
    axis::Int

    function AxisAppendBuffer(data::Array{T, N}, length::Integer, axis::Integer) where {T, N}
        1 <= axis <= N || throw(ArgumentError("axis must be between 1 and $N"))
        0 <= length <= size(data, axis) || throw(ArgumentError("length must fit within buffer capacity"))
        new{T, N}(data, Int(length), Int(axis))
    end
end

"""
    PagedAxisAppendBuffer{T,N}

Paged append buffer that stores cached sequence chunks across multiple backing
arrays instead of repeatedly reallocating one growing tensor.
"""
mutable struct PagedAxisAppendBuffer{T, N} <: AbstractAxisBuffer{T, N}
    pages::Vector{Array{T, N}}
    page_lengths::Vector{Int}
    base_shape::NTuple{N, Int}
    length::Int
    axis::Int
    page_capacity::Int

    function PagedAxisAppendBuffer(
        pages::Vector{Array{T, N}},
        page_lengths::Vector{Int},
        base_shape::NTuple{N, Int},
        total_length::Integer,
        axis::Integer,
        page_capacity::Integer,
    ) where {T, N}
        1 <= axis <= N || throw(ArgumentError("axis must be between 1 and $N"))
        length(pages) == length(page_lengths) || throw(ArgumentError("pages and page_lengths must match"))
        page_capacity > 0 || throw(ArgumentError("page_capacity must be positive"))
        used = 0
        for (page, page_len) in zip(pages, page_lengths)
            0 <= page_len <= size(page, axis) || throw(ArgumentError("page length must fit within page capacity"))
            for dim_idx in 1:N
                dim_idx == axis && continue
                size(page, dim_idx) == base_shape[dim_idx] || throw(DimensionMismatch("page shape does not match base shape outside the append axis"))
            end
            used += page_len
        end
        used == Int(total_length) || throw(ArgumentError("page lengths must sum to total length"))
        new{T, N}(pages, page_lengths, base_shape, Int(total_length), Int(axis), Int(page_capacity))
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

function PagedAxisAppendBuffer(chunk::AbstractArray{T, N}, axis::Integer=2; capacity::Union{Nothing, Integer}=nothing, page_capacity::Union{Nothing, Integer}=nothing) where {T, N}
    1 <= axis <= N || throw(ArgumentError("axis must be between 1 and $N"))
    chunk_len = size(chunk, axis)
    requested = capacity === nothing ? chunk_len : Int(capacity)
    requested >= chunk_len || throw(ArgumentError("capacity must be at least the chunk length"))
    base = ntuple(i -> size(chunk, i), N)
    page_cap = page_capacity === nothing ? max(1, chunk_len) : Int(page_capacity)
    page_cap > 0 || throw(ArgumentError("page_capacity must be positive"))
    buffer = PagedAxisAppendBuffer(Array{T, N}[], Int[], base, 0, axis, page_cap)
    ensure_axis_capacity!(buffer, requested)
    return buffer
end

"""
    filled_axis_buffer(chunk; axis=2, capacity=nothing)

Create a paged append buffer and immediately populate it with `chunk`.
"""
function filled_axis_buffer(chunk::AbstractArray{T, N}; axis::Integer=2, capacity::Union{Nothing, Integer}=nothing) where {T, N}
    buffer = PagedAxisAppendBuffer(chunk, axis; capacity=capacity)
    append_axis_buffer!(buffer, chunk)
    return buffer
end

function _buffer_ranges(buffer::AxisAppendBuffer, axis_range)
    return ntuple(i -> i == buffer.axis ? axis_range : Colon(), ndims(buffer.data))
end

function _buffer_ranges(buffer::PagedAxisAppendBuffer, axis_range)
    return ntuple(i -> i == buffer.axis ? axis_range : Colon(), length(buffer.base_shape))
end

_buffer_shape(buffer::AxisAppendBuffer) = size(buffer.data)
_buffer_shape(buffer::PagedAxisAppendBuffer) = buffer.base_shape

function _check_axis_compatible(buffer::AbstractAxisBuffer{T, N}, chunk::AbstractArray{S, N}) where {T, S, N}
    eltype(chunk) == T || throw(ArgumentError("chunk element type $(eltype(chunk)) does not match buffer element type $T"))
    for dim_idx in 1:N
        dim_idx == buffer.axis && continue
        _buffer_shape(buffer)[dim_idx] == size(chunk, dim_idx) || throw(DimensionMismatch("chunk shape does not match buffer shape outside the append axis"))
    end
    return nothing
end

"""
    ensure_axis_capacity!(buffer, required_length)

Ensure that `buffer` can hold at least `required_length` items along its append
axis without further structural growth.
"""
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

function ensure_axis_capacity!(buffer::PagedAxisAppendBuffer{T, N}, required_length::Integer) where {T, N}
    required = Int(required_length)
    required >= 0 || throw(ArgumentError("required_length must be non-negative"))
    current_capacity = buffer_capacity(buffer)
    required <= current_capacity && return buffer

    while current_capacity < required
        page_len = min(buffer.page_capacity, required - current_capacity)
        dims = collect(buffer.base_shape)
        dims[buffer.axis] = page_len
        push!(buffer.pages, Array{T, N}(undef, Tuple(dims)))
        push!(buffer.page_lengths, 0)
        current_capacity += page_len
    end
    return buffer
end

"""
    append_axis_buffer!(buffer, chunk)

Append `chunk` along the configured append axis of `buffer`.
"""
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

"""Return the total currently allocated capacity of `buffer` along its append axis."""
function buffer_capacity(buffer::AxisAppendBuffer)
    return size(buffer.data, buffer.axis)
end

"""Return the total currently allocated capacity of `buffer` along its append axis."""
function buffer_capacity(buffer::PagedAxisAppendBuffer)
    return sum(size(page, buffer.axis) for page in buffer.pages; init=0)
end

"""Return the number of backing pages used by `buffer`."""
buffer_page_count(buffer::AxisAppendBuffer) = 1
"""Return the number of backing pages used by `buffer`."""
buffer_page_count(buffer::PagedAxisAppendBuffer) = length(buffer.pages)

function _append_page!(buffer::PagedAxisAppendBuffer{T, N}, chunk::AbstractArray{T, N}, chunk_range, page_idx::Int, page_range) where {T, N}
    page = buffer.pages[page_idx]
    dest_ranges = ntuple(i -> i == buffer.axis ? page_range : Colon(), N)
    src_ranges = ntuple(i -> i == buffer.axis ? chunk_range : Colon(), N)
    @views page[dest_ranges...] .= chunk[src_ranges...]
    buffer.page_lengths[page_idx] += length(page_range)
    return buffer
end

function append_axis_buffer!(buffer::PagedAxisAppendBuffer{T, N}, chunk::AbstractArray{T, N}) where {T, N}
    _check_axis_compatible(buffer, chunk)
    chunk_len = size(chunk, buffer.axis)
    ensure_axis_capacity!(buffer, buffer.length + chunk_len)

    remaining = chunk_len
    chunk_start = 1
    page_idx = 1
    while remaining > 0
        while page_idx <= length(buffer.pages) && buffer.page_lengths[page_idx] == size(buffer.pages[page_idx], buffer.axis)
            page_idx += 1
        end
        page_idx <= length(buffer.pages) || error("paged buffer capacity tracking is inconsistent")
        available = size(buffer.pages[page_idx], buffer.axis) - buffer.page_lengths[page_idx]
        take = min(available, remaining)
        # Copy only the current slice into the active page so the cache can grow
        # across multiple pages without rebuilding one contiguous tensor.
        page_range = (buffer.page_lengths[page_idx] + 1):(buffer.page_lengths[page_idx] + take)
        chunk_range = chunk_start:(chunk_start + take - 1)
        _append_page!(buffer, chunk, chunk_range, page_idx, page_range)
        buffer.length += take
        chunk_start += take
        remaining -= take
    end
    return buffer
end

"""
    buffer_view(buffer)

Return the populated portion of `buffer` as an array-like object.

For paged buffers this may materialize a concatenated view of the used pages.
"""
function buffer_view(buffer::AxisAppendBuffer)
    ranges = _buffer_ranges(buffer, 1:buffer.length)
    return @view buffer.data[ranges...]
end

function buffer_view(buffer::PagedAxisAppendBuffer{T, N}) where {T, N}
    if isempty(buffer.pages)
        dims = collect(buffer.base_shape)
        dims[buffer.axis] = 0
        return Array{T, N}(undef, Tuple(dims))
    end
    pieces = [
        begin
            page_len = buffer.page_lengths[idx]
            page_len == 0 ? nothing : @view(buffer.pages[idx][_buffer_ranges(buffer, 1:page_len)...])
        end
        for idx in eachindex(buffer.pages)
    ]
    used = [piece for piece in pieces if piece !== nothing]
    isempty(used) && begin
        dims = collect(buffer.base_shape)
        dims[buffer.axis] = 0
        return Array{T, N}(undef, Tuple(dims))
    end
    length(used) == 1 && return used[1]
    # The write path is truly paged; read-side concatenation preserves the
    # existing cache API until page-aware attention kernels land.
    return cat(used...; dims=buffer.axis)
end

Base.:(==)(a::AbstractAxisBuffer, b::AbstractAxisBuffer) =
    a.axis == b.axis && a.length == b.length && Array(buffer_view(a)) == Array(buffer_view(b))
