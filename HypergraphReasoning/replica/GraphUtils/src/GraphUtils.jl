"""
GraphUtils.jl - Utility functions for text processing and extraction.

Replicates functionality from HyperGraphReasoningPython/GraphReasoning/utils.py
"""
module GraphUtils

using ArgParse

"""
    extract(string, start='[', end=']')

Extract substring between first occurrence of `start` and last occurrence of `end`.

# Arguments
- `string::String`: Input string to extract from
- `start::Char`: Starting delimiter (default: '[')
- `end_char::Char`: Ending delimiter (default: ']')

# Returns
- `String`: Extracted substring including delimiters, or empty string if not found

# Examples
```julia
extract("[hello] world [test]")  # Returns "[hello] world [test]"
extract("no brackets")           # Returns ""
```
"""
function extract(string::String, start::Char='[', end_char::Char=']')
    start_index = findfirst(start, string)
    end_index = findlast(end_char, string)
    
    if start_index === nothing || end_index === nothing
        return ""
    end
    
    if start_index.start > end_index.start
        return ""
    end
    
    return string[start_index.start:end_index.start]
end

"""
    remove_markdown_symbols(text)

Remove markdown formatting symbols from text, preserving content.

# Arguments
- `text::String`: Input text with markdown formatting

# Returns
- `String`: Text with markdown symbols removed

# Examples
```julia
remove_markdown_symbols("**bold** text")  # Returns "bold text"
remove_markdown_symbols("[link](url)")    # Returns "link"
```
"""
function remove_markdown_symbols(text::String)
    import Base: @__MODULE__
    using Regex
    
    # Remove links: [text](url) -> text
    text = replace(text, r"\[([^\]]+)\]\([^\)]+\)" => s"\1")
    
    # Remove images: ![alt](url) -> empty
    text = replace(text, r"!\[[^\]]*\]\([^\)]+\)" => "")
    
    # Remove headers: # text -> text
    text = replace(text, r"#+\s" => "")
    
    # Remove bold: **text** -> text
    text = replace(text, r"\*\*([^*]+)\*\*" => s"\1")
    
    # Remove italic: *text* -> text
    text = replace(text, r"\*([^*]+)\*" => s"\1")
    
    # Remove bold (underscore): __text__ -> text
    text = replace(text, r"__([^_]+)__" => s"\1")
    
    # Remove italic (underscore): _text_ -> text
    text = replace(text, r"_([^_]+)_" => s"\1")
    
    # Remove inline code: `code` -> code
    text = replace(text, r"`([^`]+)`" => s"\1")
    
    # Remove blockquotes: > text -> text
    text = replace(text, r"^>\s+"m => "")
    
    # Remove strikethrough: ~~text~~ -> text
    text = replace(text, r"~~(.*?)~~" => s"\1")
    
    # Remove code blocks: ```...``` -> empty
    text = replace(text, r"```.*?```"s => "")
    
    # Remove extra newlines
    text = replace(text, r"\n\s*\n" => "\n\n")
    
    # Remove list markers: * item, - item, + item, 1. item
    text = replace(text, r"^[\*\-\+]\s+"m => "")
    text = replace(text, r"^\d+\.\s+"m => "")
    
    return strip(text)
end

"""
    contains_phrase(main_string, phrase)

Check if phrase exists in main string.

# Arguments
- `main_string::String`: String to search in
- `phrase::String`: Phrase to search for

# Returns
- `Bool`: true if phrase found, false otherwise
"""
function contains_phrase(main_string::String, phrase::String)
    return occursin(phrase, main_string)
end

"""
    make_dir_if_needed(dir_path)

Create directory if it doesn't exist.

# Arguments
- `dir_path::String`: Directory path to create

# Returns
- `String`: Status message
"""
function make_dir_if_needed(dir_path::String)
    if !isdir(dir_path)
        mkpath(dir_path)
        return "Directory created."
    else
        return "Directory already exists."
    end
end

"""
    cli_main(args=ARGS)

Command-line interface for GraphUtils.jl.

# Arguments
- `args::Vector{String}`: Command-line arguments (default: ARGS)

# Commands
- `extract`: Extract substring between delimiters
- `remove-markdown`: Remove markdown symbols from text
"""
function cli_main(args::Vector{String}=ARGS)
    parser = ArgParseSettings(
        description="GraphUtils.jl - Text processing utilities",
        version="0.1.0",
        add_version=true
    )
    
    @add_arg_table! parser begin
        "command"
            help = "Command to execute: extract, remove-markdown"
            required = true
        "--input"
            help = "Input text (or read from stdin)"
            default = ""
        "--start"
            help = "Start delimiter for extract command (default: '[')"
            default = '['
        "--end"
            help = "End delimiter for extract command (default: ']')"
            default = ']'
        "--output"
            help = "Output file path (default: stdout)"
            default = ""
    end
    
    parsed_args = parse_args(args, parser)
    command = parsed_args["command"]
    
    # Read input
    input_text = if !isempty(parsed_args["input"])
        parsed_args["input"]
    else
        read(stdin, String)
    end
    
    # Execute command
    result = if command == "extract"
        extract(input_text, parsed_args["start"], parsed_args["end"])
    elseif command == "remove-markdown"
        remove_markdown_symbols(input_text)
    else
        error("Unknown command: $command")
    end
    
    # Write output
    if !isempty(parsed_args["output"])
        open(parsed_args["output"], "w") do f
            write(f, result)
        end
    else
        println(result)
    end
    
    return 0
end

end # module GraphUtils
