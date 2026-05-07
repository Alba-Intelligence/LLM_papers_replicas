"""
GraphGeneration.jl - Generate hypergraphs from text documents.

Replicates functionality from HyperGraphReasoningPython/GraphReasoning/graph_generation.py
"""
module GraphGeneration

using DataFrames
using Hypergraphs
using Graphs
using JSON
using JLD2
using SHA
using ArgParse
using Logging

# Include LoggingUtils (local module)
include("LoggingUtils.jl")
using .LoggingUtils

# Import GraphUtils (local dependency)
# Note: In production, this would be: using GraphUtils
# For now, we'll include it directly or use a local path
# include("../GraphUtils/src/GraphUtils.jl")
# using .GraphUtils

"""
    documents2dataframe(documents::Vector{String}) -> DataFrame

Convert vector of text documents to DataFrame with chunk_id.

# Arguments
- `documents::Vector{String}`: Vector of text strings

# Returns
- `DataFrame` with columns: `text`, `chunk_id`

# Throws
- `ArgumentError` if documents is empty
"""
function documents2dataframe(documents::Vector{String})::DataFrame
    log_info("documents2dataframe called", operation="documents2dataframe", package="GraphGeneration", document_count=length(documents))
    
    if isempty(documents)
        log_error("documents2dataframe: empty documents vector", operation="documents2dataframe", package="GraphGeneration")
        throw(ArgumentError("documents cannot be empty"))
    end
    
    rows = []
    for chunk in documents
        chunk_id = bytes2hex(sha256(chunk))
        push!(rows, (text=chunk, chunk_id=chunk_id))
    end
    
    df = DataFrame(rows)
    log_info("documents2dataframe completed", operation="documents2dataframe", package="GraphGeneration", output_rows=nrow(df))
    return df
end

"""
    recursive_character_text_splitter(text::String; chunk_size::Int=2500, chunk_overlap::Int=0) -> Vector{String}

Split text into chunks using recursive character splitting (equivalent to LangChain's RecursiveCharacterTextSplitter).

# Arguments
- `text::String`: Input text to split
- `chunk_size::Int`: Maximum chunk size in characters (default: 2500)
- `chunk_overlap::Int`: Overlap between chunks in characters (default: 0)

# Returns
- `Vector{String}`: Vector of text chunks
"""
function recursive_character_text_splitter(text::String; chunk_size::Int=2500, chunk_overlap::Int=0)::Vector{String}
    if length(text) <= chunk_size
        return [text]
    end
    
    chunks = String[]
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    start_idx = 1
    while start_idx <= length(text)
        end_idx = min(start_idx + chunk_size - 1, length(text))
        chunk_text = text[start_idx:end_idx]
        
        # Try to split at a good boundary
        if end_idx < length(text)
            # Look for last occurrence of separators
            best_split = 0
            for sep in separators
                if !isempty(sep)
                    last_occurrence = findlast(sep, chunk_text)
                    if last_occurrence !== nothing
                        best_split = last_occurrence.start + length(sep) - 1
                        break
                    end
                end
            end
            
            if best_split > 0
                chunk_text = text[start_idx:start_idx + best_split - 1]
                start_idx += best_split - chunk_overlap
            else
                start_idx = end_idx + 1 - chunk_overlap
            end
        else
            start_idx = length(text) + 1
        end
        
        push!(chunks, chunk_text)
    end
    
    return chunks
end

"""
    hypergraphPrompt(text::String, generate::Function, ...) -> Tuple{Hypergraph, DataFrame}

Extract hypergraph from text using LLM prompt (equivalent to Python's hypergraphPrompt).

# Arguments
- `text::String`: Input text
- `generate::Function`: LLM generation function (system_prompt, prompt) -> response
- `generate_figure::Union{Function, Nothing}`: Optional function for image analysis
- `image_list::Union{Vector{String}, Nothing}`: Optional vector of image paths
- `metadata::Dict`: Metadata dictionary (e.g., {"chunk_id" => "..."})
- `do_distill::Bool`: Whether to distill text first (default: true)
- `do_relabel::Bool`: Whether to use relation names as edge IDs (default: false)
- `repeat_refine::Int`: Number of refinement passes (default: 0)
- `verbatim::Bool`: Whether to print progress (default: false)

# Returns
- `Tuple{Hypergraph, DataFrame}`: Hypergraph and chunk DataFrame
"""
function hypergraphPrompt(
    text::String,
    generate::Function;
    generate_figure::Union{Function, Nothing}=nothing,
    image_list::Union{Vector{String}, Nothing}=nothing,
    metadata::Dict=Dict(),
    do_distill::Bool=true,
    do_relabel::Bool=false,
    repeat_refine::Int=0,
    verbatim::Bool=false
)::Tuple{Hypergraph, DataFrame}
    
    # System prompts (matching Python implementation)
    SYS_PROMPT_DISTILL = "You are provided with a context chunk (delimited by ```) Your task is to respond with a concise scientific heading, summary, and a bullited list to your best understanding and all of them should include reasoning. You should ignore human-names, references, or citations."
    
    USER_PROMPT_DISTILL = "In a matter-of-fact voice, rewrite this ```$text```. The writing must stand on its own and provide all background needed, and include details. Ignore references. Extract the table if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any."
    
    SYS_PROMPT_GRAPHMAKER = (
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. " *
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology of terms mentioned in the given context, representing the key concepts as per the context with well-defined and widely used names of materials, systems, methods. " *
        "You always report a technical term or abbreviation and keep it as it is. " *
        "If you receive a location to an image, you must use it as a node which <id> will be the location and the <type> will be \"image\" and relate the information in the context to make the nodes and edges relation. " *
        "<relation> in an edge must truly reveal important information that can provide scientific insight from the <source> to the <target> " *
        "Return a JSON with two fields: <nodes> and <edges>. " *
        "Each node must have <id> and <type>. " *
        "Each edge must have <source>, <target>, and <relation>."
    )
    
    USER_PROMPT = "Context: ```$text``` \n Extract the knowledge graph in structured JSON: "
    
    # Distill text if requested
    if do_distill
        if verbatim
            println("Distilling text...")
        end
        text = generate(SYS_PROMPT_DISTILL, USER_PROMPT_DISTILL)
    end
    
    # Generate hypergraph JSON
    if verbatim
        println("Generating triples...")
    end
    
    result_json = generate(SYS_PROMPT_GRAPHMAKER, USER_PROMPT)
    
    # Parse JSON response
    try
        result = JSON.parse(result_json)
        
        # Extract nodes and edges
        nodes = get(result, "nodes", [])
        edges = get(result, "edges", [])
        
        # Create hypergraph
        hg = Hypergraph()
        
        # Add nodes
        node_ids = String[]
        for node in nodes
            node_id = get(node, "id", "")
            node_type = get(node, "type", "")
            if !isempty(node_id)
                push!(node_ids, node_id)
                # Add node attributes if needed
            end
        end
        
        # Add edges (hyperedges)
        for edge in edges
            source = get(edge, "source", "")
            target = get(edge, "target", "")
            relation = get(edge, "relation", "")
            
            if !isempty(source) && !isempty(target)
                # Create hyperedge connecting source and target
                add_hyperedge!(hg, [source, target])
            end
        end
        
        # Create chunk DataFrame
        chunk_df = DataFrame(
            chunk_id = [get(metadata, "chunk_id", "")],
            text = [text]
        )
        
        return (hg, chunk_df)
        
    catch e
        @warn "Failed to parse LLM response: $e"
        # Return empty hypergraph
        return (Hypergraph(), DataFrame())
    end
end

"""
    df2hypergraph(df::DataFrame, generate::Function, ...) -> Tuple{Hypergraph, Vector{DataFrame}}

Build hypergraph from DataFrame of text chunks.

# Arguments
- `df::DataFrame`: DataFrame with `text` and `chunk_id` columns
- `generate::Function`: LLM generation function
- `generate_figure::Union{Function, Nothing}`: Optional function for image analysis
- `image_list::Union{Vector{String}, Nothing}`: Optional vector of image paths
- `do_distill::Bool`: Whether to distill text first (default: true)
- `do_relabel::Bool`: Whether to use relation names as edge IDs (default: false)
- `repeat_refine::Int`: Number of refinement passes (default: 0)
- `verbatim::Bool`: Whether to print progress (default: false)

# Returns
- `Tuple{Hypergraph, Vector{DataFrame}}`: Hypergraph and list of sub-DataFrames

# Throws
- `ArgumentError` if df missing required columns
"""
function df2hypergraph(
    df::DataFrame,
    generate::Function;
    generate_figure::Union{Function, Nothing}=nothing,
    image_list::Union{Vector{String}, Nothing}=nothing,
    do_distill::Bool=true,
    do_relabel::Bool=false,
    repeat_refine::Int=0,
    verbatim::Bool=false
)::Tuple{Union{Hypergraph, Nothing}, Vector{DataFrame}}
    
    # Validate required columns
    if !("text" in names(df)) || !("chunk_id" in names(df))
        log_error("df2hypergraph: missing required columns", operation="df2hypergraph", package="GraphGeneration", columns=names(df))
        throw(ArgumentError("DataFrame must contain 'text' and 'chunk_id' columns"))
    end
    
    log_info("df2hypergraph called", operation="df2hypergraph", package="GraphGeneration", input_rows=nrow(df))
    
    if isempty(df)
        return (nothing, DataFrame[])
    end
    
    sub_hgs = Hypergraph[]
    sub_dfs = DataFrame[]
    
    # Process each row
    for row in eachrow(df)
        try
            metadata = Dict("chunk_id" => row.chunk_id)
            
            hg, chunk_df = hypergraphPrompt(
                row.text,
                generate;
                generate_figure=generate_figure,
                image_list=image_list,
                metadata=metadata,
                do_distill=do_distill,
                do_relabel=do_relabel,
                repeat_refine=repeat_refine,
                verbatim=verbatim
            )
            
            # Only keep valid hypergraphs
            if nv(hg) > 0 || ne(hg) > 0
                push!(sub_hgs, hg)
                push!(sub_dfs, chunk_df)
            elseif verbatim
                println("Skipping chunk $(row.chunk_id) – returned empty hypergraph")
            end
            
        catch e
            if verbatim
                println("Exception while processing chunk $(row.chunk_id): $e")
            end
        end
    end
    
    if isempty(sub_hgs)
        if verbatim
            println("No valid subgraphs found. Returning nothing.")
        end
        return (nothing, DataFrame[])
    end
    
    # Union all hypergraphs
    H = Hypergraph()
    for hg in sub_hgs
        # Union operation (simplified - actual implementation may need more sophisticated merging)
        for v in vertices(hg)
            add_vertex!(H, v)
        end
        for e in hyperedges(hg)
            add_hyperedge!(H, e)
        end
    end
    
    return (H, sub_dfs)
end

"""
    make_hypergraph_from_text(txt::String, generate::Function, graph_root::String; ...) -> Tuple{String, Hypergraph, String, Vector{DataFrame}}

Build or load hypergraph from text, with caching.

# Arguments
- `txt::String`: Input text string
- `generate::Function`: LLM generation function
- `graph_root::String`: Identifier for output files
- `chunk_size::Int`: Text chunk size (default: 2500)
- `chunk_overlap::Int`: Overlap between chunks (default: 0)
- `do_distill::Bool`: Whether to distill text (default: true)
- `do_relabel::Bool`: Whether to use relation names as edge IDs (default: false)
- `data_dir::String`: Output directory (default: "data")
- `verbatim::Bool`: Whether to print progress (default: false)

# Returns
- `Tuple{String, Hypergraph, String, Vector{DataFrame}}`: (pkl_path, hypergraph, sub_dfs_pkl_path, sub_dfs)
"""
function make_hypergraph_from_text(
    txt::String,
    generate::Function,
    graph_root::String;
    chunk_size::Int=2500,
    chunk_overlap::Int=0,
    do_distill::Bool=true,
    do_relabel::Bool=false,
    data_dir::String="data",
    verbatim::Bool=false
)::Tuple{String, Hypergraph, String, Vector{DataFrame}}
    
    # Create data directory if needed
    if !isdir(data_dir)
        mkpath(data_dir)
    end
    
    # Generate file paths
    pkl_path = joinpath(data_dir, "$(graph_root).jld2")
    sub_dfs_pkl_path = joinpath(data_dir, "$(graph_root)_sub_dfs.jld2")
    
    # Check cache
    if isfile(pkl_path)
        if verbatim
            println("Loading cached hypergraph from $pkl_path")
        end
        hg = load(pkl_path, "hypergraph")
        sub_dfs = load(sub_dfs_pkl_path, "sub_dfs")
        return (pkl_path, hg, sub_dfs_pkl_path, sub_dfs)
    end
    
    # Split text into chunks
    chunks = recursive_character_text_splitter(txt; chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Convert to DataFrame
    df = documents2dataframe(chunks)
    
    # Generate hypergraph
    hg, sub_dfs = df2hypergraph(
        df, generate;
        do_distill=do_distill,
        do_relabel=do_relabel,
        verbatim=verbatim
    )
    
    if hg === nothing
        error("Failed to generate hypergraph")
    end
    
    # Save to cache
    save(pkl_path, "hypergraph", hg)
    save(sub_dfs_pkl_path, "sub_dfs", sub_dfs)
    
    return (pkl_path, hg, sub_dfs_pkl_path, sub_dfs)
end

"""
    cli_main(args=ARGS)

Command-line interface for GraphGeneration.jl.
"""
function cli_main(args::Vector{String}=ARGS)
    parser = ArgParseSettings(
        description="GraphGeneration.jl - Generate hypergraphs from text",
        version="0.1.0",
        add_version=true
    )
    
    @add_arg_table! parser begin
        "--input"
            help = "Input text file (or read from stdin)"
            default = ""
        "--output"
            help = "Output file path for hypergraph (JLD2 format)"
            default = ""
        "--format"
            help = "Output format: jld2, json, graphml"
            default = "jld2"
        "--chunk-size"
            help = "Text chunk size"
            arg_type = Int
            default = 2500
        "--chunk-overlap"
            help = "Chunk overlap"
            arg_type = Int
            default = 0
        "--graph-root"
            help = "Graph root identifier for caching"
            default = "graph"
        "--data-dir"
            help = "Data directory for caching"
            default = "data"
        "--verbose"
            help = "Verbose output"
            action = :store_true
    end
    
    parsed_args = parse_args(args, parser)
    
    # Read input
    input_text = if !isempty(parsed_args["input"])
        read(parsed_args["input"], String)
    else
        read(stdin, String)
    end
    
    # Mock generate function (in production, this would be a real LLM interface)
    function mock_generate(system_prompt::String, prompt::String)
        return """{"nodes": [{"id": "node1", "type": "concept"}], "edges": []}"""
    end
    
    # Generate hypergraph
    pkl_path, hg, sub_dfs_pkl_path, sub_dfs = make_hypergraph_from_text(
        input_text,
        mock_generate,
        parsed_args["graph-root"];
        chunk_size=parsed_args["chunk-size"],
        chunk_overlap=parsed_args["chunk-overlap"],
        data_dir=parsed_args["data-dir"],
        verbatim=parsed_args["verbose"]
    )
    
    # Write output
    if !isempty(parsed_args["output"])
        if parsed_args["format"] == "jld2"
            save(parsed_args["output"], "hypergraph", hg)
        elseif parsed_args["format"] == "json"
            # Convert to JSON format (simplified)
            open(parsed_args["output"], "w") do f
                JSON.print(f, Dict("nodes" => vertices(hg), "edges" => hyperedges(hg)))
            end
        end
    else
        # Output to stdout as JSON
        println(JSON.json(Dict(
            "pkl_path" => pkl_path,
            "nodes" => length(vertices(hg)),
            "edges" => length(hyperedges(hg))
        )))
    end
    
    return 0
end

end # module
