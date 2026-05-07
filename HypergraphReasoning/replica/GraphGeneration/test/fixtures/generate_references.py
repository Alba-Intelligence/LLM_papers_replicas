#!/usr/bin/env python3
"""
Generate reference test fixtures from Python HyperGraphReasoning package.

This script runs the Python implementation and saves outputs in formats
that can be loaded by Julia tests for equivalence comparison.
"""
import sys
import json
import pickle
from pathlib import Path

# Add HyperGraphReasoning to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "HyperGraphReasoningPython"))

try:
    from GraphReasoning import graph_generation, graph_analysis, graph_tools, utils
except ImportError as e:
    print(f"Error importing HyperGraphReasoning: {e}")
    print("Make sure HyperGraphReasoningPython is available in the parent directory")
    sys.exit(1)


def generate_documents2dataframe_reference():
    """Generate reference output for documents2dataframe function."""
    documents = [
        "This is the first document.",
        "This is the second document with more content.",
        "A third document for testing."
    ]
    
    df = graph_generation.documents2dataframe(documents)
    
    # Save as JSON (Julia can read this)
    output_path = Path(__file__).parent / "documents2dataframe_reference.json"
    df_dict = {
        "text": df["text"].tolist(),
        "chunk_id": df["chunk_id"].tolist()
    }
    
    with open(output_path, "w") as f:
        json.dump(df_dict, f, indent=2)
    
    print(f"Generated reference: {output_path}")


def generate_df2hypergraph_reference():
    """Generate reference output for df2hypergraph function."""
    # Create sample DataFrame
    import pandas as pd
    df = pd.DataFrame({
        "text": ["Sample text chunk 1", "Sample text chunk 2"],
        "chunk_id": ["chunk_1", "chunk_2"]
    })
    
    # Note: This requires LLM setup, so we'll create a minimal mock
    # In practice, you'd run the actual function with proper LLM configuration
    print("Note: df2hypergraph requires LLM setup. Create mock reference manually if needed.")


if __name__ == "__main__":
    print("Generating Python reference fixtures...")
    generate_documents2dataframe_reference()
    generate_df2hypergraph_reference()
    print("Done!")
