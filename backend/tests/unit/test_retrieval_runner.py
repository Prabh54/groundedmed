"""eval/runners/retrieval.py: dev-only inputs without labels, and summary math."""

from eval.runners.retrieval import load_dev_questions, summarize


def test_dev_questions_carry_no_answer_label():
    questions = load_dev_questions()
    assert len(questions) == 100
    assert all(set(q) == {"pmid", "question"} for q in questions)


def row(gold, dense, final, total_ms, skipped=False):
    return {
        "gold_pmid": gold,
        "dense_pmids": dense,
        "final_pmids": final,
        "rerank_skipped": skipped,
        "timings_ms": {"total": total_ms},
        "usage": [
            {"model": "voyage-4", "tokens": 10, "cost_usd": 0.001},
            {"model": "rerank-3", "tokens": 100, "cost_usd": 0.002},
        ],
    }


def test_summarize_hand_example():
    rows = [
        row("g", ["x", "g"], ["g", "x"], 100),  # dense rank 2 -> reranked rank 1
        row("g", ["g", "x"], ["x", "g"], 200),  # dense rank 1 -> reranked rank 2
        row("g", ["x", "y"], ["y", "x"], 300, skipped=True),  # never found
    ]
    s = summarize(rows)
    assert s["n_questions"] == 3
    assert s["dense"]["hit_at_1"] == 1 / 3 and s["dense"]["hit_at_5"] == 2 / 3
    assert s["reranked"]["hit_at_1"] == 1 / 3 and s["reranked"]["hit_at_5"] == 2 / 3
    assert s["rerank_skipped"] == 1
    assert s["latency_ms"]["p50"] == 200
    assert s["tokens"] == {"voyage-4": 30, "rerank-3": 300}
    assert s["list_price_usd"] == 0.009
