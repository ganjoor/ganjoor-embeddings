"""
Reads couplet-level (poem id, VOrder, CoupletIndex, CoupletSummary, verse text) entries from a
local clone of ganjoor-data, for the WHOLE corpus — not scoped to any one poet, unlike the
Hafez-ghazal pilot, which queried the live database directly (CoupletSummary wasn't part of the
public export yet at the time). Reuses iter_poem_ids_and_paths() from data_source.py directly,
same id-index walk as the poem-level pipeline.

Trusts CoupletSummary's presence as the sole signal for "this is a couplet anchor," rather than
hardcoding which VersePosition values are legitimate anchors. The Hafez pilot confirmed the
common case (VersePosition "Right"/"Left", summary on Right only) directly, and a database
breakdown found two rarer position values with high CoupletSummary coverage (95-98%) whose exact
enum names in the exported JSON aren't confirmed. Rather than guess those names, this walks every
verse in a poem and picks up any with a non-empty CoupletSummary as an anchor — correct by
construction for any structural type, known or not.

The next verse (VOrder + 1), if any, is carried along purely as optional DISPLAY context — never
part of what gets embedded (CoupletSummary alone, the same decision made for the pilot). So even
on the rare position types where "next verse" isn't a true paired partner, this can't affect
embedding quality, only a cosmetic detail if that couplet's text is ever shown somewhere.

A known, accepted, negligible limitation: ~52 verses corpus-wide are known to carry a stray
CoupletSummary on what should be the "second half" of a pair (a confirmed data anomaly, not a
second legitimate copy) — trusting presence alone means these produce a small number of
near-duplicate extra entries. At roughly 0.003% of the corpus, this isn't worth building
position-name-specific exclusion logic to prevent, especially given the exact enum names needed
for that logic aren't confirmed.
"""
import json
import os
import sys
from typing import Iterator, NamedTuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import _load_json, iter_poem_ids_and_paths  # noqa: E402


class CoupletEntry(NamedTuple):
    id: str  # f"{poem_id}:{v_order}" - a composite string id, for checkpointing (couplets don't have a single simple integer id the way poems do)
    poem_id: int
    v_order: int
    couplet_index: Optional[int]
    summary: str
    right_text: str
    left_text: Optional[str]
    full_url: str


def _read_poem_verses(ganjoor_data_root: str, full_url: str) -> Optional[list]:
    trimmed = full_url.lstrip("/")
    path = os.path.join(ganjoor_data_root, "poets", trimmed + ".json")
    if not os.path.isfile(path):
        return None

    poem = _load_json(path)
    return poem.get("Verses")


def _extract_couplets_from_verses(verses: list, poem_id: int, full_url: str) -> list:
    if not verses:
        return []

    by_vorder = {v["VOrder"]: v for v in verses}
    entries = []

    # Confirmed directly against a real corpus-wide breakdown (94.26% Right, 4.98% Paragraph,
    # 0.76% CenteredVerse1, the rest negligible anomalies) which of these position TYPES
    # legitimately pair with a second line at VOrder+1, versus which are standalone units that
    # should never be paired with whatever verse happens to follow them. "Right" pairs with
    # "Left" (the vast majority of the corpus); "CenteredVerse1" pairs with "CenteredVerse2"
    # (a much rarer, but real, second couplet-like structure). Everything else - most
    # importantly "Paragraph", prose-narrative content from works like Maqamat, confirmed by
    # its verse lengths (100-300+ chars, versus 20-35 for genuine verse halves) to be a
    # different kind of content entirely - never gets a partner: pairing it with whatever
    # unrelated verse follows was a real, confirmed display bug in the previous version of this
    # function. The anchor-detection rule itself (CoupletSummary presence alone decides what's
    # eligible at all) is unchanged - this only tightens which anchors get a second line shown.
    PARTNER_POSITION = {"Right": "Left", "CenteredVerse1": "CenteredVerse2"}

    for v in verses:
        summary = v.get("CoupletSummary")
        if not summary or not summary.strip():
            continue

        v_order = v["VOrder"]
        position = v.get("Position")
        expected_partner_position = PARTNER_POSITION.get(position)

        partner_text = None
        if expected_partner_position:
            partner = by_vorder.get(v_order + 1)
            if partner and partner.get("Position") == expected_partner_position:
                partner_text = partner["Text"]
            # if position expects a partner but the actual VOrder+1 verse isn't the right type,
            # deliberately leave partner_text as None rather than pairing with the wrong thing

        entries.append(CoupletEntry(
            id=f"{poem_id}:{v_order}",
            poem_id=poem_id,
            v_order=v_order,
            couplet_index=v.get("CoupletIndex"),
            summary=summary,
            right_text=v["Text"],
            left_text=partner_text,
            full_url=full_url,
        ))

    return entries


def iter_couplet_summaries(ganjoor_data_root: str) -> Iterator[CoupletEntry]:
    """
    The main entry point: yields CoupletEntry for every couplet (across every poem, in ascending
    poem-id order) that has a non-empty CoupletSummary. A poem with no Verses field, or whose
    file is missing entirely (shouldn't normally happen for an id that came from the id index,
    but the export and index could theoretically drift), is silently skipped, same philosophy as
    the poem-level pipeline's handling of poems with no PoemSummary.
    """
    for poem_id, full_url in iter_poem_ids_and_paths(ganjoor_data_root):
        verses = _read_poem_verses(ganjoor_data_root, full_url)
        if not verses:
            continue
        for entry in _extract_couplets_from_verses(verses, poem_id, full_url):
            yield entry
