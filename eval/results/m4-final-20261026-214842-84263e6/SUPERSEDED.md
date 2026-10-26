# Superseded — do not report from this folder

Replaced by the next `m4-final-*` folder, computed from the same inputs after the independent review
(2026-10-26). Reviewer: every value here matched an independent recomputation. Fixed afterwards:
per-stage latency included stages that did not run (stored as 0.0 after a gate refusal), so baseline
synthesis and agent synthesizer/critic showed n=200 instead of the questions where they ran (p50 about
20–40 ms low); the judge join now also checks kind, citation IDs and the blinding key; raw.jsonl now
carries token usage; methodology.md shows more digits. No headline accuracy, coverage, Hit@k, overall
latency, cost or comparison value changes. Kept on disk as the record.
