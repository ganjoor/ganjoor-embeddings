# ganjoor-embeddings

Semantic ("find a poem about...") search for [Ganjoor](https://ganjoor.net) works by comparing
vector embeddings of poem summaries against a vector embedding of what a person types. This repo
is everything needed to understand, verify, or reproduce those embeddings — the tooling that
generated them, not a trained model (nothing here was trained; see [`docs/MODEL.md`](docs/MODEL.md)
for the distinction, it matters for what's actually reproducible here).

## What's in this repo

- **`scripts/`** — the actual generation and verification tooling.
- **`tests/`** — unit and integration tests, including tests run against a real uploaded poem
  (Hafez's ghazal #1) rather than only synthetic data.
- **`docs/`** — see below.

## What's *not* in this repo

The generated embeddings themselves (`embeddings.f32` + `embeddings-index.json`, ~530MB) are
published as [GitHub Release assets](../../releases/latest) rather than committed into git —
GitHub blocks any single git-tracked file over 100MB, and a binary artifact this size doesn't
belong in commit history regardless.

## Quick start

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

## License

**GPLv3** — see [`LICENSE`](LICENSE), matching [`GanjoorService`](https://github.com/ganjoor/GanjoorService).
Applies to everything in this repo: the tooling code and the published embeddings data alike,
by deliberate choice, for consistency with the rest of the Ganjoor project family rather than
splitting code and data under different license types.
