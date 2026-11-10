"""Deterministic agent answer rules (pure Python, no LLM), shared by the critic stage and the final
validator so both apply exactly the same decision (ADR-044, ADR-045).

- `_kept` / `apply_critic`: an item survives only if its first critic result says keep + supported +
  citation_valid ("revise" counts as remove; an item without a result is removed).
- `refusal_reason`: after the citation check and the critic's removals, why the answer would be
  refused, or None. Fix B (ADR-044): no veto when the critic kept every citation-valid claim.
"""

from backend.app.guardrails.citation_validator import validate
from backend.app.models.agent import CriticResult
from backend.app.models.answer import DraftAnswer, ValidationReport

IE = "insufficient_evidence"


def _kept(result: CriticResult | None) -> bool:
    return (
        result is not None
        and result.action == "keep"
        and result.supported
        and result.citation_valid
    )


def apply_critic(draft: DraftAnswer, results: list[CriticResult]) -> tuple[DraftAnswer, list[str]]:
    """Return (draft with only kept items, removed item IDs). Never adds or edits text: the verdict
    and every kept item are the synthesizer's own."""
    by_id: dict[str, CriticResult] = {}
    for r in results:
        by_id.setdefault(r.claim_id, r)  # first result per ID wins; duplicates can't overrule it
    removed = [i.id for i in draft.claims + draft.limitations if not _kept(by_id.get(i.id))]
    cleaned = DraftAnswer(
        verdict=draft.verdict,
        claims=[c for c in draft.claims if c.id not in removed],
        limitations=[item for item in draft.limitations if item.id not in removed],
    )
    return cleaned, removed


def refusal_reason(
    draft: DraftAnswer,
    evidence_ids: list[str],
    critic_results: list[CriticResult],
    answer_supported: bool | None,
) -> tuple[str | None, DraftAnswer, ValidationReport, list[str]]:
    """(reason or None, checked draft, citation report, critic-removed IDs) for one draft."""
    validated, report = validate(draft, evidence_ids)
    checked, critic_removed = apply_critic(validated, critic_results)
    if report.downgraded_to_insufficient:
        reason = "no_supported_claims"
    elif checked.verdict != IE and not checked.claims:
        reason = "no_supported_claims"  # the critic removed every claim
    elif (
        checked.verdict != IE
        and answer_supported is False
        and len(checked.claims) < len(validated.claims)
    ):
        reason = "no_supported_claims"  # claims survive but don't support the verdict (fix B)
    elif checked.verdict == IE:
        reason = "model_abstained"
    else:
        reason = None
    return reason, checked, report, critic_removed
