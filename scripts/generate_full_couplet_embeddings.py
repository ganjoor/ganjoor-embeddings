#!/usr/bin/env python3
"""
Generates couplet-level embeddings for the WHOLE corpus, from a local ganjoor-data clone (which
must include CoupletSummary in its verse export — confirmed present as of this run, added
specifically to enable this). Unlike the Hafez-ghazal pilot (scoped to one poet, sourced from a
direct SQL export since CoupletSummary wasn't in the public export yet at the time), this reads
every poem via couplet_data_source.py, the same way generate_embeddings.py reads PoemSummary.

Checkpointed exactly like generate_embeddings.py, and for the same reason — expected corpus size
is roughly 1.5M couplets (~5.8GB), an estimated multi-day run, so surviving an interruption
(Ctrl+C, a crash, a machine reboot) via --resume matters here in a way it never did for the small,
minutes-long Hafez pilot. Checkpoint ids are composite strings ("poemId:vOrder"), not plain poem
ids, since couplets don't have a single simple integer id.

Reuses clean_summary() and build_couplet_url() from generate_couplet_pilot_embeddings.py
directly — not reimplemented — so the AI-prefix stripping and the (now-fixed) CoupletIndex+1
deep-link convention can't drift between the pilot and the full run.

Recommended order, same "verify before trusting" discipline as every other step here:
  1. --count-only         cheap, no model loaded - just confirms the extraction logic finds a
                           sane total before spending any real time or compute.
  2. --inspect-only        confirms the model's real tensor shapes (should already match the
                           pilot's — this is the same model — but free to reconfirm).
  3. --limit 20            a quick smoke test through the actual embedding pipeline.
  4. the real run          expect this to take a genuinely long time; use --resume freely.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_embeddings import (  # noqa: E402
    embed_batch,
    inspect_model,
    DEFAULT_INPUT_IDS_NAME,
    DEFAULT_ATTENTION_MASK_NAME,
    DEFAULT_OUTPUT_NAME,
)
from generate_couplet_pilot_embeddings import clean_summary, build_couplet_url  # noqa: E402
from couplet_data_source import iter_couplet_summaries  # noqa: E402


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def append_completed_ids(checkpoint_file_handle, ids):
    for entry_id in ids:
        checkpoint_file_handle.write(f"{entry_id}\n")
    checkpoint_file_handle.flush()


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_count_only(args):
    """
    No model loaded, nothing embedded - just walks the corpus and reports the total, so the
    extraction logic's real output can be sanity-checked (does the count look roughly right
    against the ~1.5M figure from the earlier database breakdown?) before spending any real time.
    """
    print(f"Walking {args.source} — this reads every poem's file once, may take a minute or two...")
    count = 0
    poems_with_couplets = set()
    start = time.time()
    for entry in iter_couplet_summaries(args.source):
        count += 1
        poems_with_couplets.add(entry.poem_id)
        if count % 100000 == 0:
            print(f"  {count} couplets so far...")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"Total couplets with a CoupletSummary: {count}")
    print(f"Distinct poems contributing at least one: {len(poems_with_couplets)}")


def run(args):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(args.model_dir, args.model_file)
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} not found")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"{tokenizer_path} not found")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    if args.inspect_only:
        inspect_model(session)
        return

    checkpoint_path = os.path.join(args.output, "checkpoint.ndjson")
    completed_ids_ordered = load_checkpoint(checkpoint_path) if args.resume else []
    completed_ids = set(completed_ids_ordered)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} couplets already embedded, skipping them.")

    print(f"Walking {args.source} for couplets with a CoupletSummary...")
    entries = list(iter_couplet_summaries(args.source))
    if args.limit:
        entries = entries[:args.limit]
    remaining = [e for e in entries if e.id not in completed_ids]
    print(f"{len(entries)} couplets have a summary; {len(remaining)} remaining to embed.")

    bin_path = os.path.join(args.output, "couplet-embeddings.f32")
    bin_mode = "ab" if (args.resume and os.path.exists(bin_path)) else "wb"
    checkpoint_mode = "a" if (args.resume and os.path.exists(checkpoint_path)) else "w"
    all_ids = list(completed_ids_ordered) if args.resume else []
    all_meta = []  # rebuilt below to stay in lockstep with all_ids

    dimension = None
    index_path = os.path.join(args.output, "couplet-embeddings-index.json")
    if args.resume and os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            old_index = json.load(f)
            dimension = old_index.get("dimension")
            all_meta = old_index.get("couplets", [])

    entries_by_id = {e.id: e for e in entries}

    start_time = time.time()
    processed_count = 0

    with open(bin_path, bin_mode) as bin_file, open(checkpoint_path, checkpoint_mode, encoding="utf-8") as checkpoint_file:
        for batch in batched(remaining, args.batch_size):
            texts = [clean_summary(e.summary) for e in batch]
            vectors = embed_batch(
                session, tokenizer, texts,
                args.input_ids_name, args.attention_mask_name, args.output_name,
                args.max_length,
            )

            if dimension is None:
                dimension = vectors.shape[1]
            elif vectors.shape[1] != dimension:
                raise RuntimeError(f"embedding dimension changed mid-run: {dimension} -> {vectors.shape[1]}")

            # same lockstep-write discipline as generate_embeddings.py - vectors and their ids
            # are recorded together, in the same iteration, so an interruption between batches
            # can never leave a vector on disk the checkpoint doesn't know about
            bin_file.write(vectors.astype(np.float32).tobytes())
            bin_file.flush()
            append_completed_ids(checkpoint_file, [e.id for e in batch])

            for e in batch:
                all_ids.append(e.id)
                completed_ids.add(e.id)
                all_meta.append({
                    "poemId": e.poem_id,
                    "vOrder": e.v_order,
                    "fullUrl": build_couplet_url(e.full_url, e.couplet_index),
                })
            processed_count += len(batch)

            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            print(f"  {len(completed_ids)}/{len(entries)} embedded ({rate:.1f} couplets/sec)")

    index = {
        "model": args.model_file,
        "dimension": dimension,
        "poolingMethod": "last_token",
        "normalized": True,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "count": len(all_meta),
        "scope": "full-corpus",
        "couplets": all_meta,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_meta)} couplets embedded, dimension={dimension}.")
    print(f"  {bin_path}")
    print(f"  {index_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="local ganjoor-data clone (must include CoupletSummary in its verse export)")
    p.add_argument("--model-dir", help="directory containing the .onnx model + tokenizer.json (not needed for --count-only)")
    p.add_argument("--model-file", default="model_quantized.onnx")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="process only the first N couplets — for a quick smoke test")
    p.add_argument("--resume", action="store_true", help="skip couplets already recorded in the checkpoint from a previous run")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--count-only", action="store_true", help="just count couplets found — no model needed, no embedding run")
    p.add_argument("--input-ids-name", default=DEFAULT_INPUT_IDS_NAME)
    p.add_argument("--attention-mask-name", default=DEFAULT_ATTENTION_MASK_NAME)
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = p.parse_args()

    if args.count_only:
        run_count_only(args)
    else:
        if not args.model_dir:
            p.error("--model-dir is required unless --count-only is set")
        run(args)
