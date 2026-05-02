"""Lightweight metadata about a tokenizer loaded through the Python bridge."""
struct PythonTokenizerHandle
    name_or_path::String
    vocab_size::Int
end

"""
    MythosTokenizer

Tokenizer wrapper that delegates to a Hugging Face tokenizer through a small
Python subprocess bridge.
"""
struct MythosTokenizer
    model_id::String
    tokenizer::PythonTokenizerHandle
    runner::Cmd
end

function _tokenizer_runner()
    if haskey(ENV, "OPENMYTHOS_TOKENIZER_PYTHON")
        return Cmd([ENV["OPENMYTHOS_TOKENIZER_PYTHON"]])
    end
    return Cmd(["uv", "run", "--with", "transformers", "python"])
end

function _run_tokenizer_python(action::String, model_id::String; payload::String="", runner::Cmd=_tokenizer_runner())
    script = """
import argparse
import base64
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--action", required=True)
parser.add_argument("--model-id", required=True)
parser.add_argument("--payload", default="")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_id)

if args.action == "metadata":
    print(base64.b64encode(tok.name_or_path.encode("utf-8")).decode("ascii"))
    print(tok.vocab_size)
elif args.action == "encode":
    text = base64.b64decode(args.payload.encode("ascii")).decode("utf-8")
    ids = tok.encode(text, add_special_tokens=False)
    print(",".join(str(i) for i in ids))
elif args.action == "decode":
    ids = [int(x) for x in args.payload.split(",") if x]
    text = tok.decode(ids, skip_special_tokens=True)
    print(base64.b64encode(text.encode("utf-8")).decode("ascii"))
else:
    raise ValueError(f"unknown action: {args.action}")
"""

    mktemp() do path, io
        write(io, script)
        close(io)
        cmd = `$runner $path --action $action --model-id $model_id --payload $payload`
        return read(cmd, String)
    end
end

function MythosTokenizer(model_id::String=DEFAULT_MODEL_ID; runner::Cmd=_tokenizer_runner())
    raw = _run_tokenizer_python("metadata", model_id; runner=runner)
    lines = split(chomp(raw), '\n')
    length(lines) == 2 || error("unexpected tokenizer metadata response")
    handle = PythonTokenizerHandle(
        String(base64decode(lines[1])),
        parse(Int, lines[2]),
    )
    return MythosTokenizer(model_id, handle, runner)
end

"""Return the vocabulary size of `tok`."""
vocab_size(tok::MythosTokenizer) = tok.tokenizer.vocab_size
"""Tokenize `text` with `tok`."""
tokenize(tok::MythosTokenizer, text::AbstractString) = encode(tok, text)
"""Decode token ids back into text with `tok`."""
detokenize(tok::MythosTokenizer, token_ids::AbstractVector{<:Integer}) = decode(tok, token_ids)

"""Encode `text` into token ids with the underlying Hugging Face tokenizer."""
function encode(tok::MythosTokenizer, text::AbstractString)
    payload = base64encode(codeunits(text))
    raw = chomp(_run_tokenizer_python("encode", tok.model_id; payload=payload, runner=tok.runner))
    isempty(raw) && return Int[]
    return parse.(Int, split(raw, ','))
end

"""Decode token ids into text with the underlying Hugging Face tokenizer."""
function decode(tok::MythosTokenizer, token_ids::AbstractVector{<:Integer})
    payload = join(token_ids, ",")
    raw = chomp(_run_tokenizer_python("decode", tok.model_id; payload=payload, runner=tok.runner))
    return String(base64decode(raw))
end
