#!/usr/bin/env python3
"""
One-time migration: converts an old-format checkpoint.json (written by an earlier version of
generate_embeddings.py, before it switched to the append-only checkpoint.ndjson format) into the
new format, so a run already in progress can be interrupted and resumed with the updated script
without losing the work already done.

Usage:
    python3 scripts/migrate_checkpoint.py --output /path/to/your/output/dir

Looks for checkpoint.json in that directory and writes checkpoint.ndjson alongside it.
Does not touch or delete the old checkpoint.json.

Ordering note: the old format stored ids as `sorted(completed_ids)`, not in the order they were
actually processed/written to embeddings.f32. That only matters if a run had already been
resumed at least once before (each resume could, in principle, add ids out of sequence relative
to the rest). For a FIRST, never-yet-resumed run, poems are always processed in ascending id
order from start to finish, so the sorted list already exactly matches the true row order —
this migration is only meant for that first-interruption case, which is what this script assumes
here. If you'd already resumed a run more than once with the old script before switching, don't
trust this migration blindly — ask before relying on it for anything more complicated.
"""
import argparse
import json
import os


def migrate(output_dir):
    old_path = os.path.join(output_dir, "checkpoint.json")
    new_path = os.path.join(output_dir, "checkpoint.ndjson")

    if not os.path.exists(old_path):
        raise FileNotFoundError(f"{old_path} not found — nothing to migrate")
    if os.path.exists(new_path):
        raise FileExistsError(f"{new_path} already exists — refusing to overwrite; delete it first if you're sure")

    with open(old_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    ids = old_data["completed_ids"]

    with open(new_path, "w", encoding="utf-8") as f:
        for poem_id in ids:
            f.write(f"{poem_id}\n")

    print(f"Migrated {len(ids)} ids from {old_path} to {new_path}.")
    print("The old checkpoint.json was left untouched (not deleted).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="the same --output directory your generate_embeddings.py run used")
    args = p.parse_args()
    migrate(args.output)
