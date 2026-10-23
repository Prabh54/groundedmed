"""Deterministic citation validator (ARCHITECTURE §7). Pure Python; no LLM.

Runs on every answer, baseline and agent:
1. Output already parsed into DraftAnswer (verdict in the allowed set) by the structured call.
2. Every claim needs >= 1 citation ID.
3. Every citation ID must match `PMID:<digits>` and be one of this request's selected evidence IDs.
   A claim with any bad citation is removed whole: an invented ID means the claim is not traceable.
4. Removed claims are counted (`removed_uncited`, `removed_invalid_citation`).
5. A yes/no/maybe verdict with zero valid claims left becomes `insufficient_evidence`; the model's
   summary and uncertainty are then dropped too (they may carry uncited or hijacked text).
Any paper a text field mentions ("PMID 123", "PubMed ID: 123") must be in the evidence set: a claim
mentioning another paper is removed as an invalid citation, a summary doing so is replaced by the
surviving claims' text, and such an uncertainty note is dropped. The free-text summary is not
claim-checked beyond that (ARCHITECTURE §7; the M3 critic covers semantics).
"""

import re

from backend.app.models.answer import Claim, DraftAnswer, ValidationReport

CITATION_ID = re.compile(r"^PMID:\d+$")
_PMID_MENTION = re.compile(r"\b(?:PMID|PubMed(?:\s*ID)?)\s*[:#]?\s*(\d+)", re.IGNORECASE)
NO_SUPPORTED_CLAIMS_SUMMARY = (
    "The retrieved evidence did not yield any citable claim for this question."
)


def mentions_unknown_paper(text: str, allowed: set[str]) -> bool:
    return any(f"PMID:{m}" not in allowed for m in _PMID_MENTION.findall(text))


def _claim_problem(claim: Claim, allowed: set[str]) -> str | None:
    if not claim.citation_ids or not claim.text.strip():
        return "uncited"
    for cid in claim.citation_ids:
        if not CITATION_ID.match(cid) or cid not in allowed:
            return "invalid"
    if mentions_unknown_paper(claim.text, allowed):
        return "invalid"
    return None


def validate(draft: DraftAnswer, evidence_ids: list[str]) -> tuple[DraftAnswer, ValidationReport]:
    """Return the cleaned answer and what was removed. Never adds content."""
    allowed = set(evidence_ids)
    report = ValidationReport()
    kept: list[Claim] = []
    for claim in draft.claims:
        problem = _claim_problem(claim, allowed)
        if problem == "uncited":
            report.removed_uncited.append(claim.id)
        elif problem == "invalid":
            report.removed_invalid_citation.append(claim.id)
        else:
            kept.append(claim)

    verdict, summary, uncertainty = draft.verdict, draft.summary, draft.uncertainty
    if verdict != "insufficient_evidence" and not kept:
        verdict, summary, uncertainty = "insufficient_evidence", NO_SUPPORTED_CLAIMS_SUMMARY, ""
        report.downgraded_to_insufficient = True
    else:
        if mentions_unknown_paper(summary, allowed):
            summary = " ".join(c.text for c in kept) or NO_SUPPORTED_CLAIMS_SUMMARY
            report.summary_replaced = True
        if mentions_unknown_paper(uncertainty, allowed):
            uncertainty = ""
            report.uncertainty_removed = True

    cleaned = DraftAnswer(verdict=verdict, summary=summary, claims=kept, uncertainty=uncertainty)
    return cleaned, report
