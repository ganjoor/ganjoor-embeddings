#!/usr/bin/env python3
"""
Sanity-checks and lets you query the Hafez-ghazal couplet-embedding pilot output — the same
spirit as verify_embeddings.py, adapted for couplet-embeddings-index.json's shape (a "couplets"
list of {poemId, vOrder, fullUrl}, not a flat "ids" list).

Basic check:
    python3 verify_couplet_pilot.py --embeddings-dir ./couplet-pilot-output

Nearest neighbors for a specific couplet (identify it by poem id + verse order, e.g. the
opening couplet of Hafez's ghazal #1 — poem id 2130, v_order 1):
    python3 verify_couplet_pilot.py --embeddings-dir ./couplet-pilot-output \
        --query-poem-id 2130 --query-vorder 1 --top-k 8
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_couplet_pilot_embeddings import _try_load_with_header, _try_load_without_header  # noqa: E402
from couplet_data_source import _read_poem_verses  # noqa: E402


def load_source_lookup(csv_path):
    """Reuses the same format-detecting loader as generation itself, so this works on whichever
    export shape (comma+header, or ؛-delimited/no-header) the CSV actually is, without needing
    to know which in advance. Returns {(poemId, vOrder): {summary, rightText, leftText}}.

    Only ever covers whatever poems the CSV itself covers (e.g. just Hafez's ghazals for the
    pilot) — once results can come from anywhere in the full corpus, most matches won't be in
    here. See --source-ganjoor-data for a lookup that works regardless of which poet a match
    happens to be from."""
    rows = _try_load_with_header(csv_path)
    if rows is None:
        rows = _try_load_without_header(csv_path, delimiter="\u061b")
    if rows is None:
        raise ValueError(f"{csv_path} doesn't match either expected export shape.")

    lookup = {}
    for row in rows:
        key = (int(row["PoemId"]), int(row["VOrder"]))
        lookup[key] = {
            "summary": row["CoupletSummary"],
            "right": row["RightText"],
            "left": row["LeftText"],
        }
    return lookup


def load_couplet_text_from_ganjoor_data(ganjoor_data_root, poem_id, v_order, full_url):
    """Reads just ONE couplet's text directly from a local ganjoor-data clone — not a full-corpus
    lookup table, just the single poem file this one couplet lives in. Cheap enough to call once
    per result (top_k is small) rather than needing to pre-build anything. Reuses
    _read_poem_verses from couplet_data_source.py, not reimplemented, so it's guaranteed to
    resolve poem file paths the exact same way generation itself did.

    Works for a couplet from ANY poet — unlike a CSV, which only ever covers whichever poems it
    was exported for."""
    base_url = full_url.split("#")[0]  # strip the #bnN anchor, that's not part of the file path
    verses = _read_poem_verses(ganjoor_data_root, base_url)
    if not verses:
        return None

    by_vorder = {v["VOrder"]: v for v in verses}
    v = by_vorder.get(v_order)
    if not v:
        return None
    partner = by_vorder.get(v_order + 1)
    return {
        "summary": v.get("CoupletSummary", ""),
        "right": v.get("Text", ""),
        "left": partner["Text"] if partner else "",
    }


def load(embeddings_dir):
    index_path = os.path.join(embeddings_dir, "couplet-embeddings-index.json")
    bin_path = os.path.join(embeddings_dir, "couplet-embeddings.f32")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    count = index["count"]
    dimension = index["dimension"]
    couplets = index["couplets"]

    expected_size = count * dimension * 4
    actual_size = os.path.getsize(bin_path)

    vectors = np.fromfile(bin_path, dtype=np.float32).reshape(count, dimension)

    return index, couplets, vectors, expected_size, actual_size


def basic_checks(index, couplets, vectors, expected_size, actual_size):
    print(f"Model: {index.get('model')}, scope: {index.get('scope')}")
    print(f"Pooling: {index.get('poolingMethod')}, normalized: {index.get('normalized')}")
    print(f"Generated: {index.get('generatedAtUtc')}")
    print(f"Count: {index['count']}, dimension: {index['dimension']}")
    print()

    ok = True

    if actual_size != expected_size:
        print(f"FAIL  file size is {actual_size} bytes, expected {expected_size} (count x dimension x 4)")
        ok = False
    else:
        print(f"OK    file size matches: {actual_size:,} bytes")

    if len(couplets) != vectors.shape[0]:
        print(f"FAIL  {len(couplets)} couplet entries but {vectors.shape[0]} vector rows")
        ok = False
    else:
        print(f"OK    couplet-entry count matches vector row count ({len(couplets)})")

    keys = [(c["poemId"], c["vOrder"]) for c in couplets]
    if len(keys) != len(set(keys)):
        dupes = len(keys) - len(set(keys))
        print(f"FAIL  {dupes} duplicate (poemId, vOrder) pair(s)")
        ok = False
    else:
        print(f"OK    no duplicate (poemId, vOrder) pairs")

    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    if nan_count or inf_count:
        print(f"FAIL  {nan_count} NaN and {inf_count} Inf values found")
        ok = False
    else:
        print("OK    no NaN/Inf values")

    sample_size = min(2000, vectors.shape[0])
    sample_idx = np.random.choice(vectors.shape[0], sample_size, replace=False)
    norms = np.linalg.norm(vectors[sample_idx], axis=1)
    bad_norms = np.sum(np.abs(norms - 1.0) > 1e-3)
    if bad_norms > 0:
        print(f"FAIL  {bad_norms}/{sample_size} sampled vectors are not unit-normalized")
        ok = False
    else:
        print(f"OK    sampled {sample_size} vectors, all unit-normalized")

    print()
    print("All checks passed." if ok else "Some checks FAILED - see above.")
    return ok


def find_similar(couplets, vectors, poem_id, v_order, top_k, source_lookup=None, ganjoor_data_root=None):
    query_idx = None
    for i, c in enumerate(couplets):
        if c["poemId"] == poem_id and c["vOrder"] == v_order:
            query_idx = i
            break

    if query_idx is None:
        print(f"\nNo couplet found with poemId={poem_id}, vOrder={v_order} in this pilot's index.")
        return

    query_vector = vectors[query_idx]
    similarities = vectors @ query_vector
    top_indices = np.argsort(-similarities)[:top_k + 1]

    def describe(c):
        entry = None
        not_found_label = ""
        if source_lookup:
            entry = source_lookup.get((c["poemId"], c["vOrder"]))
            not_found_label = "  (not found in source CSV)"
        if entry is None and ganjoor_data_root:
            entry = load_couplet_text_from_ganjoor_data(ganjoor_data_root, c["poemId"], c["vOrder"], c["fullUrl"])
            not_found_label = "  (poem file not found in ganjoor-data)"
        if entry is None:
            return not_found_label if (source_lookup or ganjoor_data_root) else ""
        return f"\n      {entry['right']}\n      {entry['left']}\n      « {entry['summary']} »"

    print(f"\nMost similar couplets to poemId={poem_id}, vOrder={v_order}:")
    for idx in top_indices:
        c = couplets[idx]
        score = similarities[idx]
        label = f"poemId={c['poemId']} vOrder={c['vOrder']} score={score:.4f}  {c['fullUrl']}"
        if idx == query_idx:
            label += "  <- this is the query couplet itself"
        print(f"  {label}{describe(c)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--query-poem-id", type=int, default=None)
    p.add_argument("--query-vorder", type=int, default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--source-csv", default=None,
                    help="a CSV export (e.g. the Hafez pilot's results.csv) — only covers "
                         "whichever poems it was exported for; prefer --source-ganjoor-data for "
                         "full-corpus results, which can come from any poet")
    p.add_argument("--source-ganjoor-data", default=None,
                    help="a local ganjoor-data clone — reads each matched couplet's text "
                         "directly, works regardless of which poet a match happens to be from. "
                         "Cheap: only reads the handful of poem files top-k actually needs, not "
                         "the whole corpus. Used as a fallback for anything --source-csv doesn't "
                         "cover, if both are given.")
    args = p.parse_args()

    index, couplets, vectors, expected_size, actual_size = load(args.embeddings_dir)
    passed = basic_checks(index, couplets, vectors, expected_size, actual_size)

    if args.query_poem_id is not None and args.query_vorder is not None:
        source_lookup = load_source_lookup(args.source_csv) if args.source_csv else None
        find_similar(couplets, vectors, args.query_poem_id, args.query_vorder, args.top_k,
                     source_lookup, args.source_ganjoor_data)
