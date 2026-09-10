#!/usr/bin/env python3
"""
Generates embeddings for every ganjoor-data poem that has a PoemSummary, using a local ONNX
export of Qwen/Qwen3-Embedding-0.6B (or any embedding model with the same last-token-pooling +
L2-normalize convention — see pooling.py's docstring).

Why this exists / design notes:
  - Query-time embedding + similarity search will run inside RMuseum via ONNX Runtime for .NET
    (Microsoft.ML.OnnxRuntime). This script MUST use the exact same ONNX model + tokenizer as
    that future .NET code — otherwise the vectors generated here and the vector generated for a
    user's search query at runtime won't live in the same embedding space, and similarity scores
    will be meaningless. Don't swap in a different model/export here without also updating the
    .NET side, and vice versa.
  - Reads from a LOCAL clone of ganjoor-data (see data_source.py) — ~130k individual file reads
    is fine on local disk, but would be slow and unkind to jsDelivr over HTTP one file at a time.
  - ~95.6% of poems have a non-empty PoemSummary as of the last check; poems without one are
    silently skipped (they just won't be reachable via semantic search until/unless a summary is
    backfilled later — a deliberate v1 scope decision, not an oversight).
  - Checkpointed: safe to interrupt (Ctrl+C, a crash, a machine reboot) and resume with --resume.
    At ~130k poems, even a fast local run is not something you want to redo from scratch after
    losing power at 90%.

Output (in --output):
  - embeddings.f32       Raw binary, N rows x D float32, row-major, sorted by poem id ascending.
                          No header — the row count and dimension are derived from
                          embeddings-index.json, so a consumer needs both files together.
  - embeddings-index.json  {"model": ..., "dimension": ..., "poolingMethod": "last_token",
                            "normalized": true, "generatedAtUtc": ..., "count": N,
                            "ids": [poem_id, ...]}   -- ids[i] corresponds to row i in the .f32 file.

Model setup (not automated by this script — download once, point --model-dir at it):
  Recommended source: https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX
  Needs, in --model-dir: the .onnx model file (+ any external-data .onnx_data file it references)
  and tokenizer.json. Confirm the actual input/output tensor names match this script's
  DEFAULT_INPUT_NAMES / DEFAULT_OUTPUT_NAME below once you have the real files — see
  inspect_model() for a quick way to check without reading this whole script.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooling import embed_pooled_and_normalized  # noqa: E402
from data_source import iter_poem_summaries  # noqa: E402

# Best-guess default tensor names for a standard ONNX export of this model family. If your
# actual model's names differ, override with --input-ids-name / --attention-mask-name /
# --output-name, or check inspect_model()'s printout when you first point this at the real files.
DEFAULT_INPUT_IDS_NAME = "input_ids"
DEFAULT_ATTENTION_MASK_NAME = "attention_mask"
DEFAULT_OUTPUT_NAME = "last_hidden_state"


def inspect_model(session):
    """Prints the ONNX graph's actual input/output names and shapes — run this once against the
    real model file (e.g. `python3 generate_embeddings.py --model-dir X --inspect-only`) before
    a full run, to confirm the DEFAULT_* names above actually match, rather than discovering a
    mismatch after the run starts failing on batch 1."""
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape} type={inp.type}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape} type={out.type}")


def load_checkpoint(checkpoint_path):
    """
    Returns poem ids in the exact order they were originally written — i.e. the same order their
    vectors appear as rows in embeddings.f32 — NOT as an unordered set. This matters: Python sets
    make no ordering guarantee (small ints often *happen* to iterate in a way that looks sorted
    under CPython, but that's an implementation detail, not something to build correctness on),
    and embeddings-index.json's "ids" list must stay aligned with embeddings.f32's actual row
    order across a --resume, or row i no longer corresponds to ids[i].
    """
    if not os.path.exists(checkpoint_path):
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return [int(line) for line in f if line.strip()]


def append_completed_ids(checkpoint_file_handle, ids):
    for poem_id in ids:
        checkpoint_file_handle.write(f"{poem_id}\n")
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


def build_extra_inputs(session, input_ids, attention_mask, input_ids_name, attention_mask_name):
    """
    This particular ONNX export is a KV-cache-enabled decoder graph (built for autoregressive
    generation), not a plain single-pass feature-extraction graph — confirmed via
    --inspect-only, which is exactly why that check exists. Its input list includes
    `position_ids` and a `past_key_values.N.key`/`.value` pair per transformer layer, even
    though we only ever want one uncached forward pass over the whole input. ONNX Runtime has
    no notion of an "optional" input at the session.run() level — every declared input must be
    fed something, or the call fails.

    This builds all of that automatically from the session's own declared input shapes, rather
    than hardcoding a specific layer/head count that would silently go stale if a different
    export variant were ever swapped in.
    """
    batch_size, seq_len = input_ids.shape
    extra = {}

    # position_ids for a fresh, uncached pass: each row's running count of real (non-padding)
    # tokens so far, minus one. Correct regardless of left- or right-padding, since it's derived
    # from that row's own attention_mask rather than assuming a fixed padding side.
    position_ids = np.clip(np.cumsum(attention_mask, axis=1) - 1, 0, None).astype(np.int64)

    for inp in session.get_inputs():
        if inp.name in (input_ids_name, attention_mask_name):
            continue
        if inp.name == "position_ids":
            extra[inp.name] = position_ids
            continue
        # past_key_values.N.key / .value: an empty (zero-length-sequence) cache tensor, built
        # from the declared shape with the real batch size substituted for the symbolic batch
        # dimension and 0 for the symbolic past-sequence-length dimension (concrete dims like
        # the head count / head dim come straight from the model's own declared shape).
        shape = []
        for dim in inp.shape:
            if isinstance(dim, int):
                shape.append(dim)
            elif dim == "batch_size":
                shape.append(batch_size)
            elif "sequence_length" in str(dim) and "past" in str(dim):
                shape.append(0)
            else:
                raise ValueError(
                    f"don't know how to size dimension '{dim}' for input '{inp.name}' "
                    f"(full shape {inp.shape}) — run --inspect-only against the real model and "
                    f"extend build_extra_inputs() to handle this export's actual input shapes"
                )
        extra[inp.name] = np.zeros(shape, dtype=np.float32)

    return extra


def embed_batch(session, tokenizer, texts, input_ids_name, attention_mask_name, output_name, max_length):
    """
    Tokenizes `texts` as a padded batch, runs the ONNX model, and returns (n_texts, dim)
    L2-normalized embeddings via last-token pooling. This is the one function that talks to both
    the tokenizer and the ONNX session — kept small and separate so embed_batch itself, not just
    its pieces, can be exercised in a test with a mock session.
    """
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    feed_dict = {
        input_ids_name: input_ids,
        attention_mask_name: attention_mask,
    }
    feed_dict.update(build_extra_inputs(session, input_ids, attention_mask, input_ids_name, attention_mask_name))

    outputs = session.run([output_name], feed_dict)
    hidden_states = outputs[0]  # (batch, seq_len, hidden_dim)

    return embed_pooled_and_normalized(hidden_states, attention_mask)


def run(args):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(args.model_dir, args.model_file)
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} not found — see this script's module docstring for where to get it")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"{tokenizer_path} not found — should ship alongside model.onnx")

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
        print(f"Resuming: {len(completed_ids)} poems already embedded, skipping them.")

    entries = list(iter_poem_summaries(args.source))
    if args.limit:
        entries = entries[:args.limit]
    remaining = [e for e in entries if e.id not in completed_ids]
    print(f"{len(entries)} poems have a summary; {len(remaining)} remaining to embed.")

    # append mode when resuming, so previously-written rows in embeddings.f32 aren't lost
    bin_path = os.path.join(args.output, "embeddings.f32")
    bin_mode = "ab" if (args.resume and os.path.exists(bin_path)) else "wb"
    # same logic for the checkpoint file - and on a FRESH (non-resume) run, explicitly reset any
    # stale checkpoint left over from an earlier attempt in this same --output dir, so it can
    # never be silently appended onto and produce a checkpoint that disagrees with a freshly
    # rewritten embeddings.f32
    checkpoint_mode = "a" if (args.resume and os.path.exists(checkpoint_path)) else "w"
    all_ids = list(completed_ids_ordered) if args.resume else []

    # if resuming and no new batches run this time (e.g. everything was already done), we must
    # not clobber the previous run's recorded dimension with None when rewriting the index below
    dimension = None
    index_path = os.path.join(args.output, "embeddings-index.json")
    if args.resume and os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            dimension = json.load(f).get("dimension")

    start_time = time.time()
    processed_count = 0

    with open(bin_path, bin_mode) as bin_file, open(checkpoint_path, checkpoint_mode, encoding="utf-8") as checkpoint_file:
        for batch in batched(remaining, args.batch_size):
            texts = [e.summary for e in batch]
            vectors = embed_batch(
                session, tokenizer, texts,
                args.input_ids_name, args.attention_mask_name, args.output_name,
                args.max_length,
            )

            if dimension is None:
                dimension = vectors.shape[1]
            elif vectors.shape[1] != dimension:
                raise RuntimeError(f"embedding dimension changed mid-run: {dimension} -> {vectors.shape[1]}")

            # write the vectors AND record their ids as done together, in the same iteration —
            # this is what keeps embeddings.f32 and checkpoint.ndjson always in lockstep, so an
            # interruption at any point between batches can never leave a vector on disk that
            # the checkpoint doesn't know about (the bug in an earlier version of this script)
            bin_file.write(vectors.astype(np.float32).tobytes())
            bin_file.flush()
            append_completed_ids(checkpoint_file, [e.id for e in batch])

            for e in batch:
                all_ids.append(e.id)
                completed_ids.add(e.id)
            processed_count += len(batch)

            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            print(f"  {len(completed_ids)}/{len(entries)} embedded ({rate:.1f} poems/sec)")

    index = {
        "model": args.model_name,
        "dimension": dimension,
        "poolingMethod": "last_token",
        "normalized": True,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "count": len(all_ids),
        "ids": all_ids,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(all_ids)} poems embedded, dimension={dimension}.")
    print(f"  {bin_path}")
    print(f"  {index_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="path to a local clone of ganjoor-data")
    p.add_argument("--model-dir", required=True, help="directory containing model.onnx + tokenizer.json")
    p.add_argument("--output", required=True, help="output directory for embeddings.f32 + embeddings-index.json")
    p.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B", help="recorded in embeddings-index.json for provenance")
    p.add_argument("--model-file", default="model.onnx", help="which .onnx file inside --model-dir to load, e.g. model_quantized.onnx for the int8 variant")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512, help="poem summaries are short paragraphs; 512 tokens is generous headroom")
    p.add_argument("--limit", type=int, default=None, help="only process the first N poems (for a quick test run)")
    p.add_argument("--resume", action="store_true", help="skip poems already recorded in the checkpoint from a previous run")
    p.add_argument("--gpu", action="store_true", help="use CUDAExecutionProvider if available")
    p.add_argument("--inspect-only", action="store_true", help="print the model's real input/output tensor names and exit, without embedding anything")
    p.add_argument("--input-ids-name", default=DEFAULT_INPUT_IDS_NAME)
    p.add_argument("--attention-mask-name", default=DEFAULT_ATTENTION_MASK_NAME)
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
