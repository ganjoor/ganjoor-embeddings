# Regenerating the embeddings

## When you'd actually want to

- **`ganjoor-data` has grown or its summaries have improved** since the currently-published
  embeddings were generated — new poems, newly-added summaries for previously-empty ones,
  corrected summaries. A full regeneration picks all of this up; there's no incremental "just
  embed the new ones" mode built in yet (see "Not built yet" below).
- **You want to try a different base model** — a newer Qwen embedding release, a different
  model family entirely, a different quantization. This is a much bigger change than it might
  look like — see "Changing the base model" below before doing this.

## Regenerating against the current model, updated `ganjoor-data`

```bash
git clone https://github.com/ganjoor/ganjoor-data.git   # or `git pull` an existing clone

python3 scripts/generate_embeddings.py \
  --source /path/to/ganjoor-data \
  --model-dir /path/to/model \
  --output ./output \
  --batch-size 16
```

Same model, same tokenizer setup as before, just against fresher source text — the safest kind
of regeneration, no re-verification of the model/tokenizer pipeline itself needed (though running
`scripts/verify_embeddings.py` on the *output* afterward is still worth doing, always).

Expect this to take a while at full corpus scale — hours, not minutes, even with a quantized
model on decent hardware; thermal throttling on some hardware (a fanless laptop, notably) can
mean the sustained rate is meaningfully slower than a short burst test suggests. Use `--limit N`
for a quick smoke test before committing to a full run, and `--resume` if it gets interrupted —
both real, tested features (see `tests/test_generate_embeddings.py`'s interruption test), not
just claimed.

## Changing the base model — a much bigger change than it looks

Swapping the model affects far more than just re-running the script:

1. **Every constant in the .NET query-time code that was confirmed via `--inspect-only` and
   reflection against THIS model's specific ONNX export** (layer count, KV-cache head
   count/dimension, the exact tokenizer construction approach, the end-of-sequence token id) is
   specific to Qwen3-Embedding-0.6B's particular export. A different model will very plausibly
   have different values for all of these, and they will NOT throw a helpful error if wrong —
   they'll either fail to compile (if you're lucky), throw a confusing runtime error, or — the
   actual failure mode this project hit once — produce output that looks plausible but is subtly
   wrong, or crash the whole process natively in a way no `try`/`catch` can stop.
2. **The dimension itself may differ** — anything hardcoded to 1024 (`EmbeddingIndex`,
   `QueryEmbedder`, `appsettings.json`'s `SemanticSearch:Dimension`) needs updating together.
3. **Redo `docs/VERIFICATION.md`'s entire checklist from scratch** against the new model —
   tokenizer parity against Python, the local console-harness crash test, all of it. None of it
   can be assumed to carry over from one model to another.

This isn't a discouragement, just an honest account of the actual scope — treat a model change
as a genuinely new integration effort, not a config flag.

## Not built yet

- **Incremental regeneration** — embedding only poems that are new or changed since the last run,
  rather than the whole corpus every time. Would meaningfully speed up routine "ganjoor-data grew
  a bit" regenerations. Not implemented; every run currently processes the full corpus (skipping
  only poems already recorded in a `--resume` checkpoint from an *interrupted* run of the *same*
  generation, not skipping poems unchanged since a previous *completed* run).
- **Backfilling the ~4.4% of poems with no `PoemSummary`** — would need summary generation first
  (a separate concern from embedding generation, and a real cost — an LLM call per poem), then
  the normal embedding pipeline on top.
