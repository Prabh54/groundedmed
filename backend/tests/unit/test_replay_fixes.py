"""M15-T6 replay rules on hand-made rows (offline, no model)."""

from eval.runners.replay_fixes import IE, published_revise_items, replay_row


def item(i, pmid="1"):
    return {"id": i, "text": f"Item {i}.", "citation_ids": [f"PMID:{pmid}"]}


def result(i, action="keep", supported=True):
    return {"claim_id": i, "supported": supported, "citation_valid": True, "action": action,
            "reason": "r"}  # fmt: skip


def row(results, answer_supported=True, claims=("c1",), verdict_now=IE):
    return {
        "abstain_reason": "no_supported_claims" if verdict_now == IE else None,
        "verdict": verdict_now,
        "evidence_pmids": ["1", "2"],
        "draft_answer": {
            "verdict": "yes",
            "claims": [item(c) for c in claims],
            "limitations": [item("l1")],
        },  # fmt: skip
        "critic_results": results,
        "answer_supported": answer_supported,
    }


def test_soften_means_delete_case_is_fixed_by_a_only():
    r = row([result("c1", "revise"), result("l1")])
    assert replay_row(r, "none") == IE and replay_row(r, "B") == IE
    assert replay_row(r, "A") == "yes" and replay_row(r, "A+B") == "yes"
    assert published_revise_items(r, "A") == 1 and published_revise_items(r, "none") == 0


def test_veto_with_every_claim_kept_is_lifted_by_b_only():
    r = row([result("c1"), result("l1")], answer_supported=False)
    assert replay_row(r, "none") == IE and replay_row(r, "A") == IE
    assert replay_row(r, "B") == "yes" and replay_row(r, "A+B") == "yes"


def test_b_keeps_the_veto_when_a_claim_was_removed():
    r = row([result("c1"), result("c2", "remove", supported=False)], answer_supported=False,
            claims=("c1", "c2"))  # fmt: skip
    assert replay_row(r, "B") == IE and replay_row(r, "A+B") == IE


def test_a_plus_b_lifts_a_veto_once_the_softened_claim_counts_as_kept():
    r = row([result("c1", "revise"), result("c2")], answer_supported=False, claims=("c1", "c2"))
    assert replay_row(r, "A") == IE and replay_row(r, "B") == IE and replay_row(r, "A+B") == "yes"


def test_unsupported_claims_are_never_kept_and_gate_refusals_never_change():
    r = row([result("c1", "revise", supported=False)])
    assert all(replay_row(r, rule) == IE for rule in ("none", "A", "B", "A+B"))
    gate = {"abstain_reason": "weak_evidence", "verdict": IE}
    assert all(replay_row(gate, rule) == IE for rule in ("none", "A", "B", "A+B"))
