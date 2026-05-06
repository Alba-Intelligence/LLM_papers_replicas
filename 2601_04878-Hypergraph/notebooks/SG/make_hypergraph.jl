### A Pluto.jl notebook ###
# ╔═╡ Cell order:
# ╠═00000000-0000-0000-0000-000000000001
# ╠═00000000-0000-0000-0000-000000000002
# ╠═00000000-0000-0000-0000-000000000003
# ╠═00000000-0000-0000-0000-000000000004
# ╠═00000000-0000-0000-0000-000000000005
# ╠═00000000-0000-0000-0000-000000000006
# ╠═00000000-0000-0000-0000-000000000007
# ╠═00000000-0000-0000-0000-000000000008
# ╠═00000000-0000-0000-0000-000000000009
# ╠═00000000-0000-0000-0000-000000000010
# ╠═00000000-0000-0000-0000-000000000011
# ╠═00000000-0000-0000-0000-000000000012
# ╠═00000000-0000-0000-0000-000000000013
# ╠═00000000-0000-0000-0000-000000000014
# ╠═00000000-0000-0000-0000-000000000015
# ╠═00000000-0000-0000-0000-000000000016
# ╠═00000000-0000-0000-0000-000000000017
# ╠═00000000-0000-0000-0000-000000000018
# ╠═00000000-0000-0000-0000-000000000019
# ╠═00000000-0000-0000-0000-000000000020
# ╠═00000000-0000-0000-0000-000000000021
# ╠═00000000-0000-0000-0000-000000000022

# ╔═╡ 00000000-0000-0000-0000-000000000001
#md """
# # Using LLMs and Knowledge graphs to search for PFAS Alternatives
# 
# ## Project with Saint Gobain
# 
# #### Yu-Chuan (Michael) Hsu, Isabella Stewart, Tarjei Hage, Wei Lu, and Markus J. Buehler, MIT, 2025
# """

# ╔═╡ 00000000-0000-0000-0000-000000000002
#md """
# # Allows for distributed or parallel processing of a dataset
# """

# ╔═╡ 00000000-0000-0000-0000-000000000003
# Thread configuration (converted from Python sys.argv handling)
thread_i = 0  # Default to single-threaded run
total_threads = 1
merge_every = 100

# ╔═╡ 00000000-0000-0000-0000-000000000004
# Configuration - LLM setup
# Note: API key should be set via environment variable
# TOGETHER_API_KEY = ENV["TOGETHER_API_KEY"]

config_list = [
    Dict(
        "model" => "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "api_key" => get(ENV, "TOGETHER_API_KEY", ""),
        "max_tokens" => 20000
    )
]

# ╔═╡ 00000000-0000-0000-0000-000000000005
# Import Julia packages (replacing Python imports)
using GraphGeneration  # Replaces: from GraphReasoning import make_hypergraph_from_text, etc.
using GraphTools       # Replaces: from GraphReasoning import generate_hypernode_embeddings, etc.
using DataFrames
using Hypergraphs
using JSON
using JLD2

# ╔═╡ 00000000-0000-0000-0000-000000000006
# Directory configuration
doc_data_dir = "./CompositePDFs_marker"
data_dir = "./GRAPHDATA_paper"
data_dir_output = "./GRAPHDATA_OUTPUT_paper"
max_tokens = config_list[1]["max_tokens"]
embedding_file = "composite_LLAMA4_70b.jld2"  # Changed from .pkl to .jld2

# ╔═╡ 00000000-0000-0000-0000-000000000007
#md """
# # Embedding the graph with Nomic
# """

# ╔═╡ 00000000-0000-0000-0000-000000000008
# Embedding generation (simplified - full implementation requires Transformers.jl integration)
if total_threads == 1
    # Note: Full embedding generation requires Transformers.jl or similar
    # This is a placeholder showing the conversion pattern
    generate_new_embeddings = true
    
    embedding_path = joinpath(data_dir, embedding_file)
    if isfile(embedding_path)
        println("Found existing embedding file")
        generate_new_embeddings = false
    end
    
    if generate_new_embeddings
        # Initialize empty hypergraph
        H = Hypergraph(Dict())
        nodes = collect(keys(H))
        
        # Generate embeddings using GraphTools.jl
        # node_embeddings = GraphTools.generate_hypernode_embeddings(nodes, embedding_model)
        # GraphTools.save_embeddings(node_embeddings, embedding_path)
    else
        # Load previously computed embeddings
        # node_embeddings = GraphTools.load_embeddings(embedding_path)
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000009
#md """
# ### Load dataset
# """

# ╔═╡ 00000000-0000-0000-0000-000000000010
# Load dataset of papers (converted from Python)
doc_list = String[]
for folder in readdir(doc_data_dir, join=false)
    md_file = joinpath(doc_data_dir, folder, "$folder.md")
    if isfile(md_file)
        push!(doc_list, md_file)
    end
end
sort!(doc_list)

# ╔═╡ 00000000-0000-0000-0000-000000000011
#md """
# ### Set up LLM client
# """

# ╔═╡ 00000000-0000-0000-0000-000000000012
# LLM client setup (requires Together.jl or similar - placeholder)
# Note: Full LLM integration requires Julia LLM packages
# client = TogetherClient(api_key=config_list[1]["api_key"])

# ╔═╡ 00000000-0000-0000-0000-000000000013
#md """
# ### Generate a Knowledge Graph (KG) from each document
# """

# ╔═╡ 00000000-0000-0000-0000-000000000014
# Initialize global hypergraph
G = Hypergraph(Dict())

# ╔═╡ 00000000-0000-0000-0000-000000000015
# Process documents and generate hypergraphs
for (i, doc) in enumerate(doc_list)
    # Only process docs for this thread
    if (i - 1) % total_threads != thread_i
        continue
    end
    
    # Extract title/doi and text
    title = replace(basename(doc), ".md" => "")
    doi = title
    
    txt = read(doc, String)
    
    # Define where this doc's subgraph lives
    graph_root = "$(i-1)_$(title[1:min(100, length(title))])"
    current_graph = joinpath(data_dir, "$graph_root.jld2")
    
    # Generate hypergraph using GraphGeneration.jl
    if !isfile(current_graph)
        println("Generating KG for $i: $title")
        try
            # Use GraphGeneration.jl function (replaces Python make_hypergraph_from_text)
            current_graph, sub_dfs = GraphGeneration.make_hypergraph_from_text(
                txt,
                graph_root=graph_root,
                chunk_size=10000,
                chunk_overlap=0,
                data_dir=data_dir
            )
            println("Generated hypergraph: $current_graph")
        catch e
            println("Error during KG generation: $e")
            sleep(60)
        end
    end
    
    # Merge into global graph (if in merging mode)
    if total_threads == 1 && i % merge_every == 0
        # Merge logic would go here
        # integrated_path, G, node_embeddings, sub_dfs = GraphGeneration.add_new_hypersubgraph_from_text(...)
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000016
#md """
# ## Summary
# 
# This notebook demonstrates the conversion from Python Jupyter to Julia Pluto.
# Key changes:
# - Python imports → Julia `using` statements
# - Python dict/list → Julia Dict/Vector
# - Python file I/O → Julia file I/O
# - GraphReasoning module → GraphGeneration.jl and GraphTools.jl packages
# - .pkl files → .jld2 files
# """

# ╔═╡ 00000000-0000-0000-0000-000000000017
# Placeholder for additional cells - full conversion requires manual refinement

# ╔═╡ 00000000-0000-0000-0000-000000000018
# Placeholder cell

# ╔═╡ 00000000-0000-0000-0000-000000000019
# Placeholder cell

# ╔═╡ 00000000-0000-0000-0000-000000000020
# Placeholder cell

# ╔═╡ 00000000-0000-0000-0000-000000000021
# Placeholder cell

# ╔═╡ 00000000-0000-0000-0000-000000000022
# Placeholder cell
