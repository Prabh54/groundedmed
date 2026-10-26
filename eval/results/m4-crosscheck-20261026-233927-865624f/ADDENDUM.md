# Addendum after the independent review (2026-10-26)

Written by hand after the reviewer's check; no number in this folder was changed.

- Every value in agreement.json, claim_support.json and report.md was recomputed independently by the
  reviewer from the raw files: all match. claim_support.json is byte-identical to the held file.
- Exact three-label agreement (rubric v2 §7) is not measurable: the second AI answered Supported /
  Not supported / Unsure (ADR-036). Later runs of crosscheck.py also state this in agreement.json.
- Agent minus baseline, claims strict: +0.0180 (95% interval -0.0004 to +0.0366). The interval only
  just includes 0, and the reviewer found the lower end turns positive (+0.0008) if questions are
  resampled in a different order. Read it as borderline, neither "a difference" nor "no difference".
- The second AI rejected 6 of the 20 claims the judge accepted (baseline 2/10, agent 4/10), mostly
  saying a conclusion was not stated in the passage. This hints that the judge's strict claim-support
  figure may be optimistic. No person checked these claims.
- Process: these figures were committed locally (baa255c) before the review and shown to the owner
  only after it.
- Code fixed after this run (no effect on these outputs): nothing is written until every input has been
  read and checked; the held file's sha256 is recorded; a wrong answer-file header is refused clearly;
  differences print with 4 decimals.
