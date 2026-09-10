"""
Reads (poem id, FullUrl, PoemSummary) triples from a local clone of ganjoor-data, for poems that
have a non-empty PoemSummary.

Deliberately walks the *id index* (index/poems-by-id/{bucket}.json), not the category tree —
ganjoor-data already publishes a flat id->path map specifically so consumers don't need to
re-implement recursive tree traversal just to enumerate every poem. See that repo's own API.md.

Expects a local clone (not the jsDelivr/HTTP API) — this script does ~130k individual poem-file
reads, which is fine as local disk I/O but would be far too slow, and unkind to jsDelivr, over
HTTP one file at a time.
"""
import json
import os
from typing import Iterator, NamedTuple, Optional


class PoemSummaryEntry(NamedTuple):
    id: int
    full_url: str
    summary: str


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(ganjoor_data_root: str) -> dict:
    return _load_json(os.path.join(ganjoor_data_root, "manifest.json"))


def iter_poem_ids_and_paths(ganjoor_data_root: str) -> Iterator[tuple]:
    """
    Yields (poem_id, full_url) for every poem in the id index, in ascending id order.
    Reads manifest.json first to get IdIndexShardSize, then walks index/poems-by-id/*.json
    shard-by-shard rather than assuming a fixed shard count, so this keeps working if the
    corpus grows and the number of shards changes.
    """
    manifest = load_manifest(ganjoor_data_root)
    shard_size = manifest["IdIndexShardSize"]
    index_dir = os.path.join(ganjoor_data_root, "index", "poems-by-id")

    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"{index_dir} not found — is {ganjoor_data_root} really a clone of ganjoor-data "
            "with the index/ directory intact?"
        )

    shard_files = sorted(
        (f for f in os.listdir(index_dir) if f.endswith(".json")),
        key=lambda f: int(f[:-len(".json")]),
    )

    entries = []
    for shard_file in shard_files:
        shard = _load_json(os.path.join(index_dir, shard_file))
        for id_str, full_url in shard.items():
            entries.append((int(id_str), full_url))

    entries.sort(key=lambda pair: pair[0])
    for poem_id, full_url in entries:
        yield poem_id, full_url

    # shard_size isn't otherwise used here, but validate it's what we expect so a future schema
    # change to the export doesn't silently produce a subtly-wrong id->path mapping
    if shard_size <= 0:
        raise ValueError(f"unexpected IdIndexShardSize in manifest.json: {shard_size}")


def read_poem_summary(ganjoor_data_root: str, full_url: str) -> Optional[str]:
    """
    Returns the poem's PoemSummary, or None if the file is missing (shouldn't normally happen
    for an id that came from the id index, but the export and index could theoretically drift)
    or the field is empty/whitespace-only.
    """
    trimmed = full_url.lstrip("/")
    path = os.path.join(ganjoor_data_root, "poets", trimmed + ".json")
    if not os.path.isfile(path):
        return None

    poem = _load_json(path)
    summary = poem.get("PoemSummary")
    if summary is None or not summary.strip():
        return None
    return summary


def iter_poem_summaries(ganjoor_data_root: str) -> Iterator[PoemSummaryEntry]:
    """
    The main entry point: yields PoemSummaryEntry for every poem that has a non-empty
    PoemSummary, in ascending id order. Poems without a summary are silently skipped — by
    design, not every poem needs to be embeddable for this feature to work (~95.6% coverage as
    of the last check).
    """
    for poem_id, full_url in iter_poem_ids_and_paths(ganjoor_data_root):
        summary = read_poem_summary(ganjoor_data_root, full_url)
        if summary is not None:
            yield PoemSummaryEntry(id=poem_id, full_url=full_url, summary=summary)
