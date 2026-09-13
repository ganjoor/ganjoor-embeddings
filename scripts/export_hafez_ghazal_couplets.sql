-- Exports couplets (with their CoupletSummary) for Hafez's ghazals specifically.
--
-- Hafez's PoetId = 2 (confirmed from the /vazn page's poet dropdown).
--
-- The CTE walks the full category subtree under Hafez's غزلیات category, the same way
-- GetDescendantCategoryIdsAsync does in C# (SemanticSearchService.cs) — not just a direct
-- match, since a "book"/collection category is rarely where poems live directly.
--
-- CoupletSummary is read ONLY from VersePosition = 0 (the "Right"/anchor verse) — per the
-- confirmed rule: it's stored once per couplet there, never legitimately on VersePosition = 1
-- (the "Left" verse); any value found there is a data anomaly, not a second copy, and this
-- query doesn't read from that position at all.
--
-- CoupletIndex is included so downstream tooling can build a deep link directly to this specific
-- couplet, not just the poem: {FullUrl}#bn{CoupletIndex} — confirmed against the real page
-- source (e.g. https://ganjoor.net/hafez/ghazal/sh1 uses #bn1, #bn2, #bn3... for its couplets in
-- order). Using this existing field rather than deriving a number from VOrder deliberately —
-- it's already computed correctly by the source system for any poem structure, including ones
-- that don't use simple Right/Left pairing throughout.
--
-- Plain SELECT (not FOR JSON PATH) — export via SSMS's "Save Results As..." → CSV, a much more
-- robust path than coaxing a single grid cell of JSON text through a clean copy/paste.

;WITH GhazalRootCat AS (
    SELECT c.Id
    FROM GanjoorCategories c
    WHERE c.PoetId = 2
      AND c.Title = N'غزلیات'
      AND c.ParentId IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM GanjoorCategories root
          WHERE root.Id = c.ParentId AND root.ParentId IS NULL
      )
),
GhazalCats AS (
    SELECT Id FROM GhazalRootCat

    UNION ALL

    SELECT child.Id
    FROM GanjoorCategories child
    INNER JOIN GhazalCats parent ON child.ParentId = parent.Id
)
SELECT
    v.PoemId,
    v.VOrder,
    v.CoupletIndex,
    v.CoupletSummary,
    v.Text AS RightText,
    vl.Text AS LeftText,
    p.FullUrl
FROM GanjoorVerses v
INNER JOIN GanjoorVerses vl
    ON vl.PoemId = v.PoemId
   AND vl.VOrder = v.VOrder + 1
   AND vl.VersePosition = 1
INNER JOIN GanjoorPoems p ON p.Id = v.PoemId
WHERE v.VersePosition = 0
  AND v.CoupletSummary IS NOT NULL
  AND v.CoupletSummary <> N''
  AND p.CatId IN (SELECT Id FROM GhazalCats)
ORDER BY v.PoemId, v.VOrder;

-- In SSMS: run this, then right-click anywhere in the results grid -> "Save Results As..." ->
-- choose "CSV (Comma delimited)" -> save as e.g. hafez_ghazal_couplets.csv. That's the file to
-- send over / point generate_couplet_pilot_embeddings.py --input at.
