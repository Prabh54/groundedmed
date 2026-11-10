# Data

Source: Hugging Face `qiaojin/PubMedQA`, pinned to revision `9001f2853fb87cab8d220904e0de81ac6973b318`
(`pqa_labeled` 1,000 rows, `pqa_unlabeled` 61,249 rows).

Rebuild from the repo root:

```bash
uv run python -m scripts.prepare_dataset
```

The script refuses to overwrite a committed manifest whose content would change, so a re-run is also a
reproducibility check: it either succeeds with identical files or stops with an error.

## Fields (ADR-009)
`long_answer` (the conclusion) and `reasoning_required_pred` / `reasoning_free_pred` are never read into
memory. `final_decision` appears only in the eval manifests below, never in the corpus.

## Manifests (`manifests/`, committed and frozen — ADR-008)

| File | Questions | yes / no / maybe |
|---|---|---|
| `test.json` | 200 | 110 / 68 / 22 |
| `dev.json` | 100 | 55 / 34 / 11 |
| `reserve.json` | 700 | 387 / 236 / 77 |
| `test2.json` (from reserve, ADR-043) | 400 | 220 / 136 / 44 |

`test.sha256` holds the SHA-256 of `test.json`; the same value is hard-coded in
`backend/tests/unit/test_manifests.py`. Check it by hand with
`cd data/manifests && shasum -a 256 -c test.sha256`.

Exact split rule: for each label in the order yes, no, maybe, sort that label's papers by PMID (as a
number), shuffle them with a fresh `random.Random(20261021)`, put the first `round(20%)` in test, the next
`round(10%)` in dev and the rest in reserve. Each manifest lists items sorted by PMID.

Reserve questions may be promoted to dev if dev proves too small (new ADR); never to test.

**test2 (ADR-043, 2026-11-10):** a second frozen test set of 400 reserve questions, locked before any further tuning
by `uv run python -m scripts.lock_test2` (excludes PMID 24519615, the ADR-025 leakage exception; per label sort by
PMID, shuffle with `random.Random(20261110)`, take 220 / 136 / 44). `test2.sha256` holds its hash, also pinned in
`backend/tests/unit/test_test2_manifest.py`. No runner loads it until its single final run. The original test set is
unchanged.
