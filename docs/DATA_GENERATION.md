# Where the data comes from, and how it was generated

## Source text: poem summaries, not poems

The embeddings are generated from each poem's `PoemSummary` field — an existing, already-public
field on every poem in [`ganjoor-data`](https://github.com/ganjoor/ganjoor-data) (Ganjoor's own
public git-tracked export). Not the raw poem text itself: a summary embeds more reliably for
"what is this poem *about*" queries than the full poem would, since a summary is already
distilled down to its actual topic/theme rather than diluted across dozens of lines of verse.

These summaries are AI-generated (visible in the data itself — they're prefixed "هوش مصنوعی:").
This project didn't generate them; it only embeds text that already existed in `ganjoor-data`.

**Coverage**: as of the run that produced the current published embeddings, 129,414 of 135,319
poems (95.6%) had a non-empty `PoemSummary`. The remaining ~4.4% are simply not searchable via
this feature — not an error, a deliberate scope decision (see `docs/REGENERATION.md` for what it
would take to backfill them).

## The generation process itself

1. Walk every poem in a local clone of `ganjoor-data` via its published id index
   (`index/poems-by-id/*.json`) — not by re-implementing category-tree traversal.
2. Skip any poem with an empty/missing `PoemSummary`.
3. For each remaining poem's summary text: tokenize, run through the ONNX model (see
   `docs/MODEL.md`), pool the last token's hidden state, L2-normalize.
4. Write the resulting vectors to `embeddings.f32` (raw float32, row-major, one row per poem, in
   ascending poem-id order) plus `embeddings-index.json` (which poem id each row corresponds to,
   plus metadata: model name, dimension, generation timestamp, pooling method).

`scripts/generate_embeddings.py` is the actual implementation; `scripts/pooling.py` and
`scripts/data_source.py` are its two real building blocks, both independently unit-tested (see
`tests/`) — `data_source.py`'s tests run against a fixture built from a real uploaded poem, not
only synthetic data.

## Is this reproducible?

Partially, and it's worth being precise about which part:

- **Deterministic, given fixed inputs**: the same summary text, run through the same exact model
  file and tokenizer configuration, produces bit-identical output every time (confirmed — see the
  determinism test in `tests/test_generate_embeddings.py`, and the "Step 4" determinism check in
  `docs/VERIFICATION.md`'s local test harness). No randomness anywhere in this pipeline.
- **Not guaranteed identical to the currently-published embeddings** if you regenerate from
  scratch today: `ganjoor-data` is a living corpus — poems get added, summaries get
  added/corrected/improved over time. Regenerating against a *different* snapshot of
  `ganjoor-data` than the one originally used will produce embeddings for a different corpus,
  not necessarily wrong, just not byte-identical to what's currently published.

## The numbers for the currently-published embeddings

- **129,414 poems** embedded, **1024 dimensions**
- `embeddings.f32`: **530,079,744 bytes** exactly (129,414 × 1024 × 4 — confirmed, not
  approximate; verify this on any copy with `scripts/verify_embeddings.py`)
- Model: `Qwen/Qwen3-Embedding-0.6B`, int8-quantized ONNX export (see `docs/MODEL.md`)

Run `scripts/verify_embeddings.py --embeddings-dir <path>` against any copy of the published
data to confirm file size, id uniqueness, absence of NaN/Inf values, and unit-normalization —
before trusting a download, not just before trusting a regeneration.
