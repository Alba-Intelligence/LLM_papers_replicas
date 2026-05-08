module TextDataCore

using BytePairEncoding
using Parquet2
import TransformerCore
using TransformerCore: text_next_token_pairs,
                       batch_next_token_pairs

include("tokenizers.jl")
include("parquet_text.jl")

export NativeTokenizerHandle,
       NativeBPETokenizer,
       vocab_size,
       tokenize,
       detokenize,
       encode,
       decode,
       vocab_texts,
       parquet_text_files,
       parquet_text_column,
       append_next_token_pairs_from_parquet!,
       next_token_batches_from_parquet

end
