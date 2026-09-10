# Verification — read this one first if you're short on time

Every check here exists because skipping it once already caused a real problem — most seriously,
a production outage. This isn't a theoretical best-practices list; it's the specific, hard-won
set of things that turned out to matter, in the order that finding each one actually happened.

## Part 1 — verifying a Python generation run

### Before running for real: inspect the actual model

```bash
python3 scripts/generate_embeddings.py --model-dir /path/to/model --inspect-only
```

Prints the ONNX model's real input/output tensor names and shapes — no embedding happens, this
just confirms the script's assumptions match the actual file. **Do this before every full run**,
not just the first time — a different model variant, or even a re-export of "the same" model, can
have different tensor names than expected.

The model this project actually uses turned out to be KV-cache-enabled (built for autoregressive
generation, not a plain single-pass embedding graph) — `position_ids` plus a `past_key_values.N.key`/
`.value` pair per layer, even though only ever used for a single uncached pass. `--inspect-only`
is what caught this before a real run, not a guess or documentation.

### Before a full run: a small `--limit` test

```bash
python3 scripts/generate_embeddings.py --source ... --model-dir ... --output ./test --limit 20
```

Confirms the whole pipeline actually works end-to-end against the real model before committing to
a run that can take hours. Check the output dimension matches what `docs/MODEL.md` says it should
be (1024) — a mismatch here means something is fundamentally wrong before you've wasted any real
time.

### After any run, regeneration or not: `verify_embeddings.py`

```bash
python3 scripts/verify_embeddings.py --embeddings-dir ./output
python3 scripts/verify_embeddings.py --embeddings-dir ./output --query-id <a real poem id> --top-k 8 --source /path/to/ganjoor-data
```

Checks file size against the expected `count × dimension × 4` byte formula exactly, duplicate
ids, NaN/Inf values, and unit-normalization. The `--query-id` mode does a real semantic sanity
check — nearest neighbors for a poem you actually know should look thematically related, not just
structurally valid. A poem's nearest neighbors all being nonsense is a real signal something is
wrong even when every other check passes.

## Part 2 — verifying a NEW consumer (a different language, a different runtime)

This is the part that actually caused a production incident, so it gets the most detail. The
existing .NET/RMuseum query-time implementation is the concrete worked example below, but every
principle generalizes to any other language building a consumer for these embeddings.

### The core risk: tokenizer parity

Whatever tokenizes a search query at query time MUST tokenize text identically to whatever
tokenized the poem summaries at generation time — same token ids, same order, same special
tokens. A mismatch here does not throw an error. It just silently produces a query embedding that
lives in a subtly different space than the document embeddings it's being compared against,
degrading result quality with nothing pointing at why.

**How this actually went wrong once**: a .NET tokenizer implementation, built from the same
`vocab.json`/`merges.txt` files, silently tokenized Persian text to **zero tokens** — every query
was reduced to nothing but a fixed instruction prefix, regardless of what was actually typed. It
compiled. It ran without error. It just didn't work, and nothing said so.

**How it was actually caught**: a direct, explicit side-by-side comparison — the same real query
strings, tokenized by the reference implementation (Python, `tokenizers.Tokenizer.from_file`) and
by the new consumer, comparing the printed token id lists directly.

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/path/to/tokenizer.json")
for text in ["a real test query", "another one", "a third, different one"]:
    ids = tok.encode(text).ids
    print(text, ids, len(ids))
```

Run the equivalent in whatever the new consumer's language is, on the exact same strings, and
diff the printed lists. Anything less than an exact match — same ids, same count, same order —
means the tokenizer isn't actually equivalent, however plausible its output looks.

**The specific thing that turned out to be missing**, discovered only through this direct
comparison, not from documentation: Python's tokenizer always appends one extra token
(`151643`, Qwen's end-of-sequence marker) after the real content. This matters beyond cosmetics —
since pooling happens on the LAST token's hidden state, a consumer missing this token pools from
a genuinely different position than the one used to generate every document embedding, even with
otherwise word-for-word identical tokenization.

### Never trust "it compiled" or "it ran" for a native-backed runtime

If the consuming language wraps a native library (ONNX Runtime's C++ core, in this project's
case), a wrong input can produce a **native crash** — not a catchable exception in the host
language, a hard process crash, invisible to any `try`/`catch` in that language. This is
genuinely what took production down once: a working local test (on different hardware, a
different OS) gave false confidence; the exact same code crashed the real production server
outright.

**Test any new native-backed consumer on a disposable local process first, on the SAME operating
system/architecture the real deployment will run on** — not just "does this compile," actually
run real inference on real queries, somewhere a crash costs nothing. Only after that succeeds
repeatedly, with real result content that looks correct, should it go anywhere near a shared or
production environment — and even then, prefer testing on an isolated/secondary deployment before
anything client-facing ever touches it.

### When the exact API surface of a library is unclear or documentation is stale

Reflection (or the equivalent introspection facility in whatever language) beats searching
documentation that might be for a different version than what's actually installed. Print the
real installed library's actual method signatures/available types directly and work from that,
rather than iterating blindly against guessed API calls — every guess that turns out wrong is a
wasted round-trip; the library itself is always the authoritative source for what's actually
callable.

### Summary checklist for any new consumer

- [ ] `--inspect-only` (or equivalent) against the real model, not assumed from documentation
- [ ] Token-id-level comparison against the Python reference tokenizer, on multiple real query
      strings, exact match required — not "looks plausible"
- [ ] A real inference call, on a disposable local process, on the actual target OS/architecture
- [ ] A determinism check — the same input should always produce the same output
- [ ] A real semantic check — embed a poem's own summary text as a "query," confirm that poem
      itself comes back as the top (or very near top) result against the published embeddings
- [ ] Only after all of the above pass: test on an isolated/non-production deployment before
      anything client-facing depends on it
