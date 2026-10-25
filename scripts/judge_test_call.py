"""M4-T0: one paid test call to the independent judge, on made-up data only.

Run: uv run python -m scripts.judge_test_call --approval "<the owner's approval, word for word>"

Uses the frozen judge prompt and schema (rubric v2) on three fictional items, so no project, dev or
test data is sent. Two free calls first (model metadata, token count); then it refuses unless the
worst case stays within --ceiling (default $0.02), counting the thinking budget ON TOP of the
output cap in case Google does not include thinking in the cap. Writes eval/results/judge-
testcall-<UTC time>/result.json: model string, Google's reported model version and metadata, rubric
fingerprints, request hash, raw response, token usage, cost, format check.
"""

import argparse
import json
from datetime import UTC, datetime

from backend.app.config import REPO_ROOT
from eval import judge_client as jc
from eval import judge_rubric as jr

MAX_OUTPUT_TOKENS = 1000
THINKING_BUDGET = 512
FAKE_QUESTION = (
    "Does drinking green tea improve word recall in adult volunteers? (fictional test record)"
)
FAKE_PASSAGE = (
    "OBJECTIVE: Fictional test record for checking a review tool; not a real study.\n"
    "METHODS: 40 adult volunteers were randomly assigned to drink green tea or water daily for "
    "four weeks, then took a 20-word recall test.\n"
    "RESULTS: The tea group recalled a mean of 14 words and the water group 11 words (p = 0.03)."
)
FAKE_ITEMS = {  # system label -> [(item id, statement)]; expected labels in the comments
    "SYSTEM_A": [("A.c1", "Volunteers who drank green tea recalled more words than those who drank "
                          "water (mean 14 vs 11, p = 0.03).")],  # supported
    "SYSTEM_B": [("B.c1", "Green tea halved the risk of dementia."),  # unsupported
                 ("B.l1", "The study was small (40 adults) and was done in Japan.")],  # partial
}  # fmt: skip


def render() -> tuple[str, str, list[str]]:
    s = jr.load_prompt()
    passage = s["PASSAGE_TEMPLATE"].format(citation_id="PMID:0000001", passage_text=FAKE_PASSAGE)
    blocks, ids = [], []
    for label, items in FAKE_ITEMS.items():
        rendered = [s["ITEM_TEMPLATE"].format(item_id=i, statement=text, passages=passage)
                    for i, text in items]  # fmt: skip
        ids += [i for i, _ in items]
        blocks.append(
            s["SYSTEM_BLOCK_TEMPLATE"].format(system_label=label, items="\n".join(rendered))
        )
    user = s["USER_TEMPLATE"].format(
        question=FAKE_QUESTION, item_ids=", ".join(ids), system_blocks="\n\n".join(blocks)
    )
    return s["SYSTEM_INSTRUCTION"], user, ids


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approval", required=True, help="the owner's approval text, word for word"
    )
    parser.add_argument("--ceiling", type=float, default=0.02, help="hard USD ceiling for the call")
    args = parser.parse_args(argv)
    if len(args.approval.strip()) < 10:
        raise SystemExit("--approval must quote the owner's approval.")
    jr.check_frozen()
    system_instruction, user, ids = render()
    body = jc.build_body(
        system_instruction, user, jr.load_schema(), temperature=0, seed=jr.SEED,
        max_output_tokens=MAX_OUTPUT_TOKENS, thinking_budget=THINKING_BUDGET,
    )  # fmt: skip

    metadata = jc.model_metadata()  # free
    input_tokens = jc.count_tokens(body)  # free
    worst = jc.judge_cost(input_tokens, MAX_OUTPUT_TOKENS + THINKING_BUDGET)
    print(f"model {jc.JUDGE_MODEL} (Google version {metadata['version']}); input {input_tokens} "
          f"tokens; worst case ${worst:.4f}; ceiling ${args.ceiling:.2f}")  # fmt: skip
    if metadata["version"] != jc.EXPECTED_MODEL_VERSION:
        raise SystemExit(f"Google now reports version {metadata['version']!r}; refusing.")
    if worst > args.ceiling:
        raise SystemExit("Worst case exceeds the ceiling; refusing before the paid call.")

    call = jc.generate(body)  # the one paid call
    raw = call["response"]
    finish = raw.get("candidates", [{}])[0].get("finishReason")
    try:
        parsed = json.loads(jc.response_text(raw))
        problems = jr.check_response(parsed, ids)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        parsed, problems = None, [f"no parseable JSON: {type(exc).__name__}"]

    out = REPO_ROOT / "eval" / "results" / f"judge-testcall-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "purpose": "M4-T0 judge test call on fictional data (no project, dev or test data)",
        "owner_approval": args.approval.strip(),
        "ceiling_usd": args.ceiling,
        "worst_case_usd": round(worst, 6),
        "judge_provider": jc.JUDGE_PROVIDER,
        "model_requested": call["model_requested"],
        "model_version_reported": call["model_version_reported"],
        "model_metadata": metadata,
        "rubric_version": jr.JUDGE_VERSION,
        "rubric_file_sha256": jr.file_hashes(),
        "generation_config": body["generationConfig"]
        | {"responseJsonSchema": "judge_v1_schema.json"},
        "request_sha256": call["request_sha256"],
        "input_tokens_counted_before": input_tokens,
        "usage": call["usage"],
        "cost_usd": round(call["cost_usd"], 6),
        "finish_reason": finish,
        "format_accepted_by_google": True,  # an HTTP error would have stopped the script above
        "format_problems": problems,
        "parsed": parsed,
        "raw_response": raw,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model_version_reported", "usage", "cost_usd",
                                              "finish_reason", "format_problems", "parsed")},
                     indent=2))  # fmt: skip
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
