# The base model

## What it is

**[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** — a 0.6B-parameter
text embedding model from Alibaba's Qwen team, decoder-only architecture, multilingual (100+
languages), 1024-dimensional output, last-token pooling.

**License: Apache 2.0** — confirmed directly on the model card, free for research and commercial
use. Not authored by this project; used as-is, unmodified weights.

## Which exact export/variant this project uses

Not the raw PyTorch weights — an ONNX export, for use with ONNX Runtime (both the Python
generation script and the .NET query-time code use ONNX Runtime, deliberately the same runtime on
both sides — see `docs/DATA_GENERATION.md` for why that specifically matters):

- **Export source**: [`onnx-community/Qwen3-Embedding-0.6B-ONNX`](https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX)
- **Variant used**: the **int8-quantized** file (`onnx/model_quantized.onnx`), not fp32. Chosen
  after benchmarking — cosine similarity between fp32 and int8 outputs on real queries averaged
  0.9997, negligible practical difference, with a meaningful speed advantage at generation time.
- **Tokenizer files**: `vocab.json` + `merges.txt` from the same export (NOT loaded via a direct
  `tokenizer.json` read — see `docs/VERIFICATION.md` for why that distinction turned out to
  matter a lot in practice).

If you're regenerating and want a different precision (fp32 for maximum fidelity, or a different
quantization), that's a one-flag change (`--model-file`) — just re-run the verification steps in
`docs/VERIFICATION.md` against whichever variant you pick, don't assume it behaves identically.

## Conventions anything consuming these embeddings must match

Get any of these wrong and similarity scores become meaningless garbage, silently — no error,
just bad results:

- **Dimension**: 1024
- **Pooling**: last non-padding token's hidden state (NOT mean pooling — this is a decoder-style
  model, mean pooling would be wrong for it)
- **Normalization**: L2-normalized, so cosine similarity between any two embeddings reduces to a
  plain dot product
- **A trailing end-of-sequence token** (id `151643`) is part of every embedded sequence — this
  comes from the tokenizer's own post-processing, not something added manually, and it changes
  which token position actually gets pooled. Skipping it produces a genuinely different (wrong)
  embedding for the same text — see `docs/VERIFICATION.md` for how this was discovered.
- **Documents (poem summaries) get no instruction prefix. Queries do** — Qwen3-Embedding's
  documented convention is asymmetric: an instruction prefix on the query side measurably
  improves retrieval, and should NOT be applied to the document/corpus side. If you ever change
  this on one side, change it on the other, or query and document embeddings drift into subtly
  different spaces.
