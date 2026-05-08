function parquet_text_files(path::AbstractString)
    if isfile(path)
        return [path]
    elseif isdir(path)
        files = String[]
        for (root, _, names) in walkdir(path)
            for name in sort(names)
                endswith(name, ".parquet") && push!(files, joinpath(root, name))
            end
        end
        return files
    end
    throw(ArgumentError("parquet path does not exist: $(path)"))
end

function parquet_text_column(path::AbstractString; column::AbstractString="text")
    ds = Parquet2.Dataset(path)
    try
        return String.(Parquet2.load(ds, column))
    catch err
        throw(ArgumentError("parquet file $(path) could not load column $(repr(column)): $(err)"))
    end
end

function _drain_next_token_pairs!(pairs, buffer::Vector{Int}, seq_len::Int, max_pairs::Int)
    while length(buffer) >= seq_len + 1 && length(pairs) < max_pairs
        chunk = buffer[1:(seq_len + 1)]
        push!(pairs, (copy(chunk[1:end-1]), copy(chunk[2:end])))
        deleteat!(buffer, 1:(seq_len + 1))
    end
    return pairs
end

function _append_next_token_pairs_from_parquet_file!(
    pairs,
    buffer::Vector{Int},
    tokenizer::NativeBPETokenizer,
    parquet_file::AbstractString,
    seq_len::Int,
    max_pairs::Int;
    column::AbstractString="text",
)
    ds = Parquet2.Dataset(parquet_file)
    try
        for row_group in ds
            texts = String.(Parquet2.load(row_group, column))
            for text in texts
                append!(buffer, encode(tokenizer, text))
                _drain_next_token_pairs!(pairs, buffer, seq_len, max_pairs)
                length(pairs) >= max_pairs && return pairs
            end
        end
    catch err
        throw(ArgumentError("parquet file $(parquet_file) could not load column $(repr(column)): $(err)"))
    end
    return pairs
end

"""
    append_next_token_pairs_from_parquet!(pairs, buffer, tokenizer, parquet_path, seq_len, max_pairs; column="text")

Incrementally extend `pairs` from local parquet shard(s), preserving any token
carry in `buffer` across files and row groups.
"""
function append_next_token_pairs_from_parquet!(
    pairs,
    buffer::Vector{Int},
    tokenizer::NativeBPETokenizer,
    parquet_path::AbstractString,
    seq_len::Integer,
    max_pairs::Integer;
    column::AbstractString="text",
)
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    max_pairs >= 0 || throw(ArgumentError("max_pairs must be non-negative"))

    for file in parquet_text_files(parquet_path)
        _append_next_token_pairs_from_parquet_file!(pairs, buffer, tokenizer, file, Int(seq_len), Int(max_pairs); column=column)
        length(pairs) >= max_pairs && break
    end
    return pairs
end

"""
    next_token_batches_from_parquet(tokenizer, parquet_path, seq_len, batch_size; max_batches=8, column="text")

Build small next-token training batches from local parquet shard(s) using a
Julia-native parquet reader and tokenizer path.
"""
function next_token_batches_from_parquet(
    tokenizer::NativeBPETokenizer,
    parquet_path::AbstractString,
    seq_len::Integer,
    batch_size::Integer;
    max_batches::Integer=8,
    column::AbstractString="text",
)
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    max_batches > 0 || throw(ArgumentError("max_batches must be positive"))

    max_pairs = Int(max_batches) * Int(batch_size)
    pairs = Tuple{Vector{Int}, Vector{Int}}[]
    buffer = Int[]
    append_next_token_pairs_from_parquet!(pairs, buffer, tokenizer, parquet_path, seq_len, max_pairs; column=column)
    return batch_next_token_pairs(pairs, batch_size; drop_last=false)
end
