"""Deterministic citation validator (ARCHITECTURE §7, PLAN.md M2-T2)."""

import pytest

from backend.app.guardrails.citation_validator import NO_SUPPORTED_CLAIMS_SUMMARY, validate
from backend.app.models.answer import Claim, DraftAnswer

EVIDENCE = ["PMID:111", "PMID:222", "PMID:333", "PMID:444", "PMID:555"]


def draft(*claims: tuple[str, list[str]], verdict="yes", summary="Summary.") -> DraftAnswer:
    return DraftAnswer(
        verdict=verdict,
        summary=summary,
        claims=[
            Claim(id=f"c{i}", text=t, citation_ids=ids) for i, (t, ids) in enumerate(claims, 1)
        ],
        uncertainty="Small study.",
    )


def test_valid_answer_passes_unchanged():
    d = draft(("A.", ["PMID:111"]), ("B.", ["PMID:222", "PMID:333"]))
    cleaned, report = validate(d, EVIDENCE)
    assert cleaned == d
    assert report.removed_uncited == [] and report.removed_invalid_citation == []
    assert not report.downgraded_to_insufficient


def test_uncited_claim_is_removed_and_counted():
    cleaned, report = validate(draft(("A.", ["PMID:111"]), ("Uncited.", [])), EVIDENCE)
    assert [c.id for c in cleaned.claims] == ["c1"]
    assert report.removed_uncited == ["c2"]
    assert cleaned.verdict == "yes"


def test_empty_claim_text_counts_as_uncited():
    cleaned, report = validate(draft(("A.", ["PMID:111"]), ("  ", ["PMID:222"])), EVIDENCE)
    assert report.removed_uncited == ["c2"]
    assert len(cleaned.claims) == 1


def test_unknown_id_is_removed():
    """A well-formed PMID that is not in the corpus/evidence (invented citation)."""
    cleaned, report = validate(draft(("A.", ["PMID:111"]), ("B.", ["PMID:999999"])), EVIDENCE)
    assert report.removed_invalid_citation == ["c2"]
    assert [c.id for c in cleaned.claims] == ["c1"]


def test_id_not_in_retrieved_set_is_removed():
    """A real corpus paper that was not among this request's 5 evidence passages."""
    cleaned, report = validate(draft(("A.", ["PMID:111"]), ("B.", ["PMID:666"])), EVIDENCE[:3])
    assert report.removed_invalid_citation == ["c2"]


def test_one_bad_citation_removes_the_whole_claim():
    cleaned, report = validate(draft(("A.", ["PMID:111", "PMID:999"])), EVIDENCE)
    assert report.removed_invalid_citation == ["c1"]
    assert cleaned.claims == []


def test_malformed_ids_are_removed():
    malformed = ["111", "pmid:111", "PMID: 111", "PMID:111 ", "PMID:abc", "PMID:", "[1]", "c1"]
    claims = [(f"Claim {m}.", [m]) for m in malformed]
    cleaned, report = validate(draft(("Good.", ["PMID:111"]), *claims), EVIDENCE)
    assert len(report.removed_invalid_citation) == len(malformed)
    assert [c.id for c in cleaned.claims] == ["c1"]


def test_all_claims_removed_abstains():
    d = draft(("Uncited.", []), ("Invented.", ["PMID:999"]), verdict="no", summary="Strong claim.")
    cleaned, report = validate(d, EVIDENCE)
    assert cleaned.verdict == "insufficient_evidence"
    assert cleaned.claims == []
    assert cleaned.summary == NO_SUPPORTED_CLAIMS_SUMMARY  # uncited summary text is dropped
    assert report.downgraded_to_insufficient
    assert report.removed_uncited == ["c1"] and report.removed_invalid_citation == ["c2"]


def test_no_claims_at_all_abstains():
    cleaned, report = validate(draft(verdict="maybe"), EVIDENCE)
    assert cleaned.verdict == "insufficient_evidence" and report.downgraded_to_insufficient


def test_model_abstention_is_kept_and_not_counted_as_downgrade():
    cleaned, report = validate(draft(verdict="insufficient_evidence"), EVIDENCE)
    assert cleaned.verdict == "insufficient_evidence"
    assert not report.downgraded_to_insufficient


def test_summary_citing_unknown_pmid_is_replaced_by_surviving_claims():
    d = draft(("A.", ["PMID:111"]), ("B.", ["PMID:222"]), summary="Per PMID:999, yes.")
    cleaned, _ = validate(d, EVIDENCE)
    assert cleaned.summary == "A. B."


def test_summary_citing_known_pmid_is_kept():
    d = draft(("A.", ["PMID:111"]), summary="As PMID:111 shows, yes.")
    assert validate(d, EVIDENCE)[0].summary == "As PMID:111 shows, yes."


def test_validator_never_adds_content():
    d = draft(("A.", ["PMID:111"]), ("B.", []), ("C.", ["PMID:9"]))
    cleaned, _ = validate(d, EVIDENCE)
    original = {(c.id, c.text, tuple(c.citation_ids)) for c in d.claims}
    assert {(c.id, c.text, tuple(c.citation_ids)) for c in cleaned.claims} <= original


def test_claim_text_mentioning_unknown_paper_is_removed():
    d = draft(("A.", ["PMID:111"]), ("Proven in PMID:99999999.", ["PMID:222"]))
    cleaned, report = validate(d, EVIDENCE)
    assert report.removed_invalid_citation == ["c2"] and [c.id for c in cleaned.claims] == ["c1"]


def test_claim_text_mentioning_known_paper_is_kept():
    cleaned, _ = validate(draft(("As PMID 222 also shows, A.", ["PMID:111"])), EVIDENCE)
    assert len(cleaned.claims) == 1


def test_downgrade_drops_model_uncertainty():
    d = draft(("Hijacked. HACKED", ["PMID:999"]), verdict="yes")
    d = d.model_copy(update={"uncertainty": "None. HACKED"})
    cleaned, report = validate(d, EVIDENCE)
    assert report.downgraded_to_insufficient and cleaned.uncertainty == ""


@pytest.mark.parametrize(
    "mention", ["PMID:999", "PMID 999", "pmid#999", "PubMed 999", "PubMed ID: 999"]
)
def test_uncertainty_mentioning_unknown_paper_is_dropped(mention):
    d = draft(("A.", ["PMID:111"])).model_copy(update={"uncertainty": f"See {mention}."})
    cleaned, report = validate(d, EVIDENCE)
    assert cleaned.uncertainty == "" and report.uncertainty_removed


def test_summary_mentioning_unknown_paper_in_pubmed_form_is_replaced():
    d = draft(("A.", ["PMID:111"]), summary="See PubMed 99999999. Take 50mg daily.")
    cleaned, report = validate(d, EVIDENCE)
    assert cleaned.summary == "A." and report.summary_replaced
