#!/usr/bin/env python3
"""
Generates couplet-level embeddings for a small, deliberate pilot: Hafez's ghazals specifically,
using CoupletSummary (a per-couplet summary field) instead of the poem-level PoemSummary that
scripts/generate_embeddings.py uses.

Why a separate, small pilot rather than the full corpus:
  - CoupletSummary is NOT part of the public ganjoor-data export schema (confirmed absent from
    every DTO this project built) — unlike PoemSummary, this data has to come directly from the
    database, not a ganjoor-data clone. See export_hafez_ghazal_couplets.sql.
  - Full couplet-level embeddings for the whole corpus would be ~1.5M rows (~5.8GB, an estimated
    5-14 days of generation time based on measured throughput) — a genuinely large commitment.
    This pilot validates the whole approach — extraction, generation, and eventually the .NET
    consuming side — on a small, deliberately chosen slice first: Hafez's ghazals, the exact
    case (loosely-connected couplets within one poem) that motivated this feature in the first
    place.

Reuses embed_batch() from generate_embeddings.py directly (not reimplemented) — the same ONNX
session setup, KV-cache tensor construction, and pooling/normalization, so this pilot's
embeddings are guaranteed to live in the exact same space as the poem-level ones, not a subtly
different one from a parallel implementation drifting out of sync.

Input: a CSV file exported from SSMS ("Save Results As..." -> CSV) after running
export_hafez_ghazal_couplets.sql — columns: PoemId, VOrder, CoupletSummary, RightText, LeftText,
FullUrl.

Output (in --output), same conventions as embeddings.f32/-index.json, just at couplet
granularity:
  - couplet-embeddings.f32        Raw binary, N rows x D float32, row-major, in input file order.
  - couplet-embeddings-index.json {"model", "dimension", "poolingMethod": "last_token",
                                    "normalized": true, "generatedAtUtc", "count", "scope",
                                    "couplets": [{"poemId", "vOrder", "fullUrl"}, ...]}
                                    — couplets[i] corresponds to row i in the .f32 file.

Embeds CoupletSummary alone (not concatenated with the raw verse text) — same convention as
embedding PoemSummary rather than raw poem text: a clean, modern-language summary embeds more
reliably than archaic, metaphorical verse text.
"""
import argparse
import csv
import json
import os
import sys
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


EXPECTED_COLUMNS = ["PoemId", "VOrder", "CoupletIndex", "CoupletSummary", "RightText", "LeftText", "FullUrl"]

AI_GENERATED_SUMMARY_PREFIX = "هوش مصنوعی:"


def clean_summary(summary):
    """
    Strips the "هوش مصنوعی:" prefix ganjoor-data's own editing workflow requires removing once a
    human has reviewed/edited a summary — that prefix describes PROVENANCE, not MEANING, and
    embedding it verbatim would pollute the vector for every still-AI-generated summary (most of
    them) with a shared, content-irrelevant phrase. Fixed here from the start, unlike the
    poem-level PoemSummary pipeline, which still embeds this prefix verbatim — a known, separate
    issue that would need a full regeneration to fix there, not something to repeat here in new
    work just because it wasn't caught the first time.
    """
    s = summary.strip()
    if s.startswith(AI_GENERATED_SUMMARY_PREFIX):
        s = s[len(AI_GENERATED_SUMMARY_PREFIX):].strip()
    return s


def build_couplet_url(full_url, couplet_index):
    """
    A deep link to this SPECIFIC couplet, not just the poem — confirmed against the real page
    source (e.g. https://ganjoor.net/hafez/ghazal/sh1 uses #bn1, #bn2, #bn3... in couplet order).
    Falls back to the bare poem URL if CoupletIndex is missing/empty for some row, rather than
    guessing at a number.

    CoupletIndex itself is 0-indexed in the database (confirmed from a real exported value: the
    very first couplet of ghazal #1 has CoupletIndex=0), while the URL fragment is 1-indexed
    (#bn1 for that same couplet, not #bn0) — hence the +1. Caught only after actually running
    this against real data and checking the output against the live page, not from the earlier
    reasoning alone, which only had a formula-derived guess to go on, not a real value to verify.
    """
    if couplet_index is None or str(couplet_index).strip() == "":
        return full_url
    return f"{full_url}#bn{int(couplet_index) + 1}"


def _try_load_with_header(input_path):
    """Standard comma-delimited CSV with a header row matching EXPECTED_COLUMNS. Returns row
    dicts, or None if this doesn't look like that shape."""
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(EXPECTED_COLUMNS).issubset(set(reader.fieldnames)):
            return None
        return list(reader)


def _try_load_without_header(input_path, delimiter):
    """A delimited CSV with NO header row, columns in EXPECTED_COLUMNS order — the shape SSMS
    produces on at least one real Persian-locale Windows/SSMS setup observed here, which used
    '؛' (the Persian/Arabic semicolon, U+061B — the OS's regional list separator) instead of a
    plain comma, with no header row either. This MUST go through a real CSV parser with that
    delimiter, not a naive str.split('؛') — CoupletSummary text itself very often contains that
    same character as ordinary Persian punctuation, and only a real CSV parser (respecting the
    quoting SSMS already applied around any field containing the delimiter) tells an internal
    punctuation mark apart from an actual field boundary."""
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))

    if not rows or any(len(r) != len(EXPECTED_COLUMNS) for r in rows):
        return None
    return [dict(zip(EXPECTED_COLUMNS, r)) for r in rows]


def load_couplets(input_path):
    """
    Reads the export from export_hafez_ghazal_couplets.sql. Tries a standard comma-delimited
    CSV with a header row first; falls back to a header-less, '؛'-delimited export if that
    doesn't match — see _try_load_without_header for why. Whichever shape is detected is printed,
    so it's visible which one actually matched rather than silently guessing.
    """
    rows = _try_load_with_header(input_path)
    mode = "comma-delimited, header row"

    if rows is None:
        rows = _try_load_without_header(input_path, delimiter="\u061b")
        mode = "؛-delimited (Persian regional separator), no header row"

    if rows is None:
        raise ValueError(
            f"{input_path} doesn't match either expected export shape (comma-delimited with a "
            f"header row, or ؛-delimited with none) — was this really exported from "
            f"export_hafez_ghazal_couplets.sql via SSMS's CSV export?"
        )

    print(f"Detected export format: {mode}")

    couplets = []
    skipped = 0
    for row in rows:
        summary = row.get("CoupletSummary")
        if summary is None or not summary.strip():
            # shouldn't happen given the SQL already filters on this, but don't trust a
            # data file blindly just because it's supposed to already be filtered
            skipped += 1
            continue
        couplets.append({
            "poem_id": int(row["PoemId"]),
            "v_order": int(row["VOrder"]),
            "summary": clean_summary(summary),
            "full_url": build_couplet_url(row["FullUrl"], row.get("CoupletIndex")),
        })

    if skipped:
        print(f"Note: skipped {skipped} rows with an empty CoupletSummary despite the SQL filter — worth a look.")
    return couplets


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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

    couplets = load_couplets(args.input)
    print(f"Loaded {len(couplets)} couplets with CoupletSummary from {args.input}")
    if args.limit:
        couplets = couplets[:args.limit]
        print(f"--limit set: processing only the first {len(couplets)}")

    if not couplets:
        print("Nothing to embed.")
        return

    all_vectors = []
    all_meta = []

    for batch in batched(couplets, args.batch_size):
        texts = [c["summary"] for c in batch]
        vectors = embed_batch(
            session, tokenizer, texts,
            args.input_ids_name, args.attention_mask_name, args.output_name,
            args.max_length,
        )
        all_vectors.append(vectors)
        all_meta.extend(
            {"poemId": c["poem_id"], "vOrder": c["v_order"], "fullUrl": c["full_url"]}
            for c in batch
        )
        print(f"  {len(all_meta)}/{len(couplets)} couplets embedded", end="\r")

    print()

    stacked = np.concatenate(all_vectors, axis=0)
    dimension = stacked.shape[1]

    bin_path = os.path.join(args.output, "couplet-embeddings.f32")
    stacked.astype(np.float32).tofile(bin_path)

    index = {
        "model": args.model_file,
        "dimension": dimension,
        "poolingMethod": "last_token",
        "normalized": True,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "count": len(all_meta),
        "scope": "hafez-ghazals-pilot",
        "couplets": all_meta,
    }
    index_path = os.path.join(args.output, "couplet-embeddings-index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(all_meta)} couplets embedded, dimension={dimension}.")
    print(f"  {bin_path}")
    print(f"  {index_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV file exported from SSMS after running export_hafez_ghazal_couplets.sql")
    p.add_argument("--model-dir", required=True, help="directory containing the .onnx model + tokenizer.json")
    p.add_argument("--model-file", default="model_quantized.onnx",
                    help="defaults to the quantized variant, matching what production actually uses "
                         "for the poem-level embeddings — keep this the same as that run unless you "
                         "have a specific reason to diverge")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512, help="couplet summaries are short; 512 is generous headroom")
    p.add_argument("--limit", type=int, default=None, help="process only the first N couplets — for a quick smoke test")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--input-ids-name", default=DEFAULT_INPUT_IDS_NAME)
    p.add_argument("--attention-mask-name", default=DEFAULT_ATTENTION_MASK_NAME)
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = p.parse_args()

    run(args)
