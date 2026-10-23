"""Deterministic citation validator (ARCHITECTURE §7). Pure Python; no LLM.

Runs on every answer, baseline and agent:
1. Output already parsed into DraftAnswer (verdict in the allowed set) by the structured call.
2. Every claim needs >= 1 citation ID.
3. Every citation ID must match `PMID:<digits>` and be one of this request's selected evidence IDs,
   and the claim's text may not mention any other paper ("PMID 123", "PubMed ID: 123").
   A claim failing any check is removed whole: an invented ID means the claim is not traceable.
4. Removed claims are counted (`removed_uncited`, `removed_invalid_citation`).
5. A yes/no/maybe verdict with zero valid claims left becomes `insufficient_evidence`, and its
   limitations are dropped too.
Limitations go through the same checks (`removed_limitations`) but never change the verdict.
Item IDs are model-written and reach the API, so they must be short and unique (`c1`, `c2`... for
claims, `l1`... for limitations); other IDs remove the item (`removed_bad_id`, by position only).
There is no free-text summary: the displayed answer is built from the surviving claims (ADR-030).
"""

import re

from backend.app.models.answer import Claim, DraftAnswer, ValidationReport

# Used with fullmatch: `$` alone would also accept a trailing newline ("PMID:1\n").
CITATION_ID = re.compile(r"PMID:\d+")
ITEM_ID = {"claims": re.compile(r"c\d{1,2}"), "limitations": re.compile(r"l\d{1,2}")}
_PMID_MENTION = re.compile(r"\b(?:PMID|PubMed(?:\s*ID)?)\s*[:#]?\s*(\d+)", re.IGNORECASE)


def mentions_unknown_paper(text: str, allowed: set[str]) -> bool:
    return any(f"PMID:{m}" not in allowed for m in _PMID_MENTION.findall(text))


def _claim_problem(claim: Claim, allowed: set[str]) -> str | None:
    if not claim.citation_ids or not claim.text.strip():
        return "uncited"
    for cid in claim.citation_ids:
        if not CITATION_ID.fullmatch(cid) or cid not in allowed:
            return "invalid"
    if mentions_unknown_paper(claim.text, allowed):
        return "invalid"
    return None


def validate(draft: DraftAnswer, evidence_ids: list[str]) -> tuple[DraftAnswer, ValidationReport]:
    """Return the cleaned answer and what was removed. Never adds or edits content."""
    allowed = set(evidence_ids)
    report = ValidationReport()

    def well_named(field: str) -> list[Claim]:
        seen: set[str] = set()
        out = []
        for i, item in enumerate(getattr(draft, field)):
            if not ITEM_ID[field].fullmatch(item.id) or item.id in seen:
                report.removed_bad_id.append(f"{field}[{i}]")  # never echo a bad ID's text
            else:
                seen.add(item.id)
                out.append(item)
        return out

    kept: list[Claim] = []
    for claim in well_named("claims"):
        problem = _claim_problem(claim, allowed)
        if problem == "uncited":
            report.removed_uncited.append(claim.id)
        elif problem == "invalid":
            report.removed_invalid_citation.append(claim.id)
        else:
            kept.append(claim)
    limitations = []
    for item in well_named("limitations"):
        if _claim_problem(item, allowed):
            report.removed_limitations.append(item.id)
        else:
            limitations.append(item)

    verdict = draft.verdict
    if verdict != "insufficient_evidence" and not kept:
        verdict, limitations = "insufficient_evidence", []
        report.downgraded_to_insufficient = True
    return DraftAnswer(verdict=verdict, claims=kept, limitations=limitations), report
