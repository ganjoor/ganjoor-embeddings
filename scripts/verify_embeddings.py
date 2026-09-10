#!/usr/bin/env python3
"""
Sanity-checks a finished embeddings.f32 + embeddings-index.json, and optionally runs a quick
nearest-neighbor query against them — a lightweight way to eyeball whether the embeddings are
actually semantically sensible before building anything further on top of them.

Basic check:
    python3 scripts/verify_embeddings.py --embeddings-dir ./output

Nearest neighbors for a specific poem (e.g. Hafez's ghazal #1, id 2130):
    python3 scripts/verify_embeddings.py --embeddings-dir ./output --query-id 2130 --top-k 8 --source /path/to/ganjoor-data

(--source is optional — without it you just get ids; with it, you get titles too, by reading the
corresponding poem.json files from a local ganjoor-data clone.)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import _load_json  # noqa: E402 - reusing the same JSON loader, not re-implementing it


def load_embeddings(embeddings_dir):
    index_path = os.path.join(embeddings_dir, "embeddings-index.json")
    bin_path = os.path.join(embeddings_dir, "embeddings.f32")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    count = index["count"]
    dimension = index["dimension"]
    ids = index["ids"]

    expected_size = count * dimension * 4
    actual_size = os.path.getsize(bin_path)

    vectors = np.fromfile(bin_path, dtype=np.float32)
    vectors = vectors.reshape(count, dimension)

    return index, ids, vectors, expected_size, actual_size


def basic_checks(index, ids, vectors, expected_size, actual_size):
    print(f"Model: {index.get('model')}")
    print(f"Pooling: {index.get('poolingMethod')}, normalized: {index.get('normalized')}")
    print(f"Generated: {index.get('generatedAtUtc')}")
    print(f"Count: {index['count']}, dimension: {index['dimension']}")
    print()

    ok = True

    if actual_size != expected_size:
        print(f"FAIL  file size is {actual_size} bytes, expected {expected_size} (count x dimension x 4)")
        ok = False
    else:
        print(f"OK    file size matches: {actual_size:,} bytes ({actual_size / 1024 / 1024:.1f} MB)")

    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        print(f"FAIL  {dupes} duplicate id(s) in embeddings-index.json")
        ok = False
    else:
        print(f"OK    no duplicate ids ({len(ids)} unique)")

    if len(ids) != vectors.shape[0]:
        print(f"FAIL  {len(ids)} ids but {vectors.shape[0]} vector rows — these must match")
        ok = False
    else:
        print(f"OK    ids count matches vector row count ({len(ids)})")

    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    if nan_count or inf_count:
        print(f"FAIL  {nan_count} NaN and {inf_count} Inf values found in vectors")
        ok = False
    else:
        print("OK    no NaN/Inf values")

    # spot-check norms on a sample rather than all 129k+ rows, for speed - a real problem would
    # show up in a random sample just as reliably as checking every row
    sample_size = min(2000, vectors.shape[0])
    sample_idx = np.random.choice(vectors.shape[0], sample_size, replace=False)
    norms = np.linalg.norm(vectors[sample_idx], axis=1)
    bad_norms = np.sum(np.abs(norms - 1.0) > 1e-3)
    if bad_norms > 0:
        print(f"FAIL  {bad_norms}/{sample_size} sampled vectors are not unit-normalized (expected all norms ~1.0)")
        print(f"      norm range in sample: {norms.min():.4f} to {norms.max():.4f}")
        ok = False
    else:
        print(f"OK    sampled {sample_size} vectors, all unit-normalized (norm ~1.0)")

    print()
    print("All checks passed." if ok else "Some checks FAILED - see above.")
    return ok


def find_similar(ids, vectors, query_id, top_k, source_dir=None):
    if query_id not in ids:
        print(f"\npoem id {query_id} not found in this embeddings set (it may not have had a PoemSummary)")
        return

    query_idx = ids.index(query_id)
    query_vector = vectors[query_idx]

    # vectors are already L2-normalized, so cosine similarity is just the dot product
    similarities = vectors @ query_vector
    top_indices = np.argsort(-similarities)[:top_k + 1]  # +1 since the query poem itself will be in there

    print(f"\nMost similar poems to id {query_id}:")
    for idx in top_indices:
        poem_id = ids[idx]
        score = similarities[idx]
        label = f"id={poem_id} score={score:.4f}"
        if poem_id == query_id:
            label += "  <- this is the query poem itself"
        elif source_dir:
            title = _lookup_title(source_dir, poem_id)
            if title:
                label += f"  {title}"
        print(f"  {label}")


def _lookup_title(source_dir, poem_id):
    """Slow (linear scan of the id index) but fine for looking up a handful of results
    interactively - not meant for anything performance-sensitive."""
    index_dir = os.path.join(source_dir, "index", "poems-by-id")
    if not os.path.isdir(index_dir):
        return None
    for shard_file in os.listdir(index_dir):
        shard = _load_json(os.path.join(index_dir, shard_file))
        if str(poem_id) in shard:
            full_url = shard[str(poem_id)]
            poem_path = os.path.join(source_dir, "poets", full_url.lstrip("/") + ".json")
            if os.path.isfile(poem_path):
                poem = _load_json(poem_path)
                return f"{poem.get('FullTitle', poem.get('Title', ''))} ({full_url})"
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--query-id", type=int, default=None, help="poem id to find nearest neighbors for")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--source", default=None, help="local ganjoor-data clone, to show titles instead of just ids")
    args = p.parse_args()

    index, ids, vectors, expected_size, actual_size = load_embeddings(args.embeddings_dir)
    passed = basic_checks(index, ids, vectors, expected_size, actual_size)

    if args.query_id is not None:
        find_similar(ids, vectors, args.query_id, args.top_k, args.source)
