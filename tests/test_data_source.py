import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/scripts")
from data_source import iter_poem_ids_and_paths, read_poem_summary, iter_poem_summaries  # noqa: E402

FIXTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ganjoor-data-fixture")


def test_iter_poem_ids_and_paths_reads_id_index():
    result = list(iter_poem_ids_and_paths(FIXTURE_ROOT))
    assert result == [(2130, "/hafez/ghazal/sh1"), (2131, "/hafez/ghazal/sh2")], f"got {result}"
    print("test_iter_poem_ids_and_paths_reads_id_index: PASS")


def test_read_poem_summary_real_hafez_poem():
    summary = read_poem_summary(FIXTURE_ROOT, "/hafez/ghazal/sh1")
    assert summary is not None
    assert summary.startswith("در این شعر حافظ از عشق و چالش‌های آن سخن می‌گوید")
    assert "پیر مغان" in summary
    print("test_read_poem_summary_real_hafez_poem: PASS")


def test_read_poem_summary_empty_returns_none():
    summary = read_poem_summary(FIXTURE_ROOT, "/hafez/ghazal/sh2")
    assert summary is None
    print("test_read_poem_summary_empty_returns_none: PASS")


def test_read_poem_summary_missing_file_returns_none():
    summary = read_poem_summary(FIXTURE_ROOT, "/hafez/ghazal/sh999-does-not-exist")
    assert summary is None
    print("test_read_poem_summary_missing_file_returns_none: PASS")


def test_iter_poem_summaries_skips_empty_and_yields_real_one():
    results = list(iter_poem_summaries(FIXTURE_ROOT))
    # sh2 has an empty summary and should be skipped entirely; only sh1 should come through
    assert len(results) == 1, f"expected exactly 1 entry (sh2 should be skipped), got {len(results)}"
    entry = results[0]
    assert entry.id == 2130
    assert entry.full_url == "/hafez/ghazal/sh1"
    assert "پیر مغان" in entry.summary
    print("test_iter_poem_summaries_skips_empty_and_yields_real_one: PASS")


if __name__ == "__main__":
    test_iter_poem_ids_and_paths_reads_id_index()
    test_read_poem_summary_real_hafez_poem()
    test_read_poem_summary_empty_returns_none()
    test_read_poem_summary_missing_file_returns_none()
    test_iter_poem_summaries_skips_empty_and_yields_real_one()
    print("\nAll data_source tests passed.")
