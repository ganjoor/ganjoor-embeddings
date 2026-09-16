"""One-off diagnostic: how much of the corpus's couplet-eligible verses are "Paragraph" type
versus the normal "Right" (and any other) type? Doesn't modify couplet_data_source.py's
CoupletEntry shape - reuses the same poem-iteration primitive, just tallies raw Position values
directly rather than committing to a permanent new field."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import _load_json, iter_poem_ids_and_paths  # noqa: E402


def run(ganjoor_data_root):
    position_counts = Counter()
    poems_with_paragraph = set()
    total = 0

    for poem_id, full_url in iter_poem_ids_and_paths(ganjoor_data_root):
        path = os.path.join(ganjoor_data_root, "poets", full_url.lstrip("/") + ".json")
        if not os.path.isfile(path):
            continue
        poem = _load_json(path)
        verses = poem.get("Verses")
        if not verses:
            continue
        for v in verses:
            summary = v.get("CoupletSummary")
            if not summary or not summary.strip():
                continue
            total += 1
            position = v.get("Position", "(none)")
            position_counts[position] += 1
            if position == "Paragraph":
                poems_with_paragraph.add(poem_id)

    print(f"Total couplet-eligible verses: {total}")
    print()
    print("Breakdown by Position value:")
    for position, count in position_counts.most_common():
        pct = 100 * count / total
        print(f"  {position:20s} {count:>10,}  ({pct:.2f}%)")
    print()
    print(f"Distinct poems contributing at least one 'Paragraph' entry: {len(poems_with_paragraph)}")


if __name__ == "__main__":
    run(sys.argv[1])
