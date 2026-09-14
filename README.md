# ganjoor-embeddings

Semantic ("find a poem about...") search for [Ganjoor](https://ganjoor.net) works by comparing
vector embeddings of poem summaries against a vector embedding of what a person types. This repo
is everything needed to understand, verify, or reproduce those embeddings — the tooling that
generated them, not a trained model (nothing here was trained; see [`docs/MODEL.md`](docs/MODEL.md)
for the distinction, it matters for what's actually reproducible here).

A second, complementary granularity also lives here: **couplet-level** embeddings, built from
each couplet's own `CoupletSummary` rather than a whole poem's `PoemSummary`. Useful specifically
where a poem's couplets don't share one coherent theme (ghazals especially) — see
[Couplet-level embeddings](#couplet-level-embeddings) below.

## What's in this repo

- **`scripts/`** — generation and verification tooling for both granularities: poem-level
  (`generate_embeddings.py`, `verify_embeddings.py`) and couplet-level
  (`generate_couplet_pilot_embeddings.py`, `generate_full_couplet_embeddings.py`,
  `couplet_data_source.py`, `verify_couplet_pilot.py`, `export_hafez_ghazal_couplets.sql`).
- **`tests/`** — unit and integration tests, including tests run against a real uploaded poem
  (Hafez's ghazal #1) rather than only synthetic data.
- **`docs/`** — see below.

## What's *not* in this repo

The generated poem-level embeddings (`embeddings.f32` + `embeddings-index.json`, ~530MB) are
published as [GitHub Release assets](../../releases/latest) rather than committed into git —
GitHub blocks any single git-tracked file over 100MB, and a binary artifact this size doesn't
belong in commit history regardless. Couplet-level embeddings, at either scope, aren't published
as release assets yet.

There's also no dedicated `docs/` file for the couplet-level scripts yet — the two sections below
are what exists for now.

## Quick start (poem-level)

```bash
pip install -r requirements.txt

# confirm the real model's actual tensor names before running anything for real -
# don't skip this, see docs/VERIFICATION.md for why
python3 scripts/generate_embeddings.py --model-dir /path/to/model --inspect-only

python3 scripts/generate_embeddings.py \
  --source /path/to/local/ganjoor-data-clone \
  --model-dir /path/to/model \
  --output ./output
```

## Couplet-level embeddings

Two separate scripts, for two different scopes — don't confuse them:

### Pilot (one poet, one form — Hafez's ghazals)

Built first, deliberately small, to validate the whole approach before committing to the full
corpus. **Requires direct read access to Ganjoor's production database** (via
`export_hafez_ghazal_couplets.sql`) — `CoupletSummary` wasn't part of the public `ganjoor-data`
export when this pilot was built, so this step isn't reproducible outside the project's own
infrastructure. (It has been added to the public export since — see the full-corpus section
below, which *is* reproducible by anyone.)

```bash
# run export_hafez_ghazal_couplets.sql against the database, export results as CSV
# (SSMS: right-click the results grid -> "Save Results As..." -> CSV)

python3 scripts/generate_couplet_pilot_embeddings.py \
  --input /path/to/exported.csv \
  --model-dir /path/to/model \
  --output ./couplet-pilot-output \
  --inspect-only

python3 scripts/generate_couplet_pilot_embeddings.py \
  --input /path/to/exported.csv \
  --model-dir /path/to/model \
  --output ./couplet-pilot-output

python3 scripts/verify_couplet_pilot.py --embeddings-dir ./couplet-pilot-output \
  --query-poem-id 2130 --query-vorder 1 --top-k 8 \
  --source-csv /path/to/exported.csv
```

### Full corpus

Reads `CoupletSummary` directly from a local `ganjoor-data` clone (no database access needed) —
reproducible by anyone. Expect roughly 1.5M couplets and a genuinely multi-day run; checkpointed
the same way `generate_embeddings.py` is, so `--resume` picks back up cleanly after any
interruption.

```bash
# cheap, no model needed - sanity-check the real count before spending any real time
python3 scripts/generate_full_couplet_embeddings.py \
  --source /path/to/local/ganjoor-data-clone \
  --output ./full-couplet-output \
  --count-only

python3 scripts/generate_full_couplet_embeddings.py \
  --source /path/to/local/ganjoor-data-clone \
  --model-dir /path/to/model \
  --output ./full-couplet-output \
  --inspect-only

python3 scripts/generate_full_couplet_embeddings.py \
  --source /path/to/local/ganjoor-data-clone \
  --model-dir /path/to/model \
  --output ./full-couplet-output \
  --limit 20

# the real run - resumable if interrupted
python3 scripts/generate_full_couplet_embeddings.py \
  --source /path/to/local/ganjoor-data-clone \
  --model-dir /path/to/model \
  --output ./full-couplet-output
# ... if interrupted:
python3 scripts/generate_full_couplet_embeddings.py \
  --source /path/to/local/ganjoor-data-clone \
  --model-dir /path/to/model \
  --output ./full-couplet-output \
  --resume

python3 scripts/verify_couplet_pilot.py --embeddings-dir ./full-couplet-output
```

`verify_couplet_pilot.py` works on either scope's output — the file format is identical, just a
different `scope` value in the index metadata (`hafez-ghazals-pilot` vs `full-corpus`).

## Documentation

- **[`docs/MODEL.md`](docs/MODEL.md)** — exactly which base model, which variant, its license,
  and the pooling/normalization convention anything consuming these embeddings must match.
- **[`docs/DATA_GENERATION.md`](docs/DATA_GENERATION.md)** — where the source text comes from,
  how the embeddings were actually produced, and what's genuinely reproducible about that.
- **[`docs/REGENERATION.md`](docs/REGENERATION.md)** — how to regenerate after the source corpus
  changes, or against a different base model.
- **[`docs/VERIFICATION.md`](docs/VERIFICATION.md)** — the specific checks worth running before
  trusting a newly-generated (or regenerated) set of embeddings. Written the hard way, after a
  real production incident — read this one before the others if you're in a hurry.

(These four predate the couplet-level scripts and describe the poem-level pipeline specifically —
still accurate for that part, just not yet extended to cover couplet-level generation.)

## License

**GPLv3** — see [`LICENSE`](LICENSE), matching [`GanjoorService`](https://github.com/ganjoor/GanjoorService).
Applies to everything in this repo: the tooling code and the published embeddings data alike,
by deliberate choice, for consistency with the rest of the Ganjoor project family rather than
splitting code and data under different license types.
