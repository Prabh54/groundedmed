"""Independent judge runner (M4-T2; rubric eval/rubric/judge_v2.md; ARCHITECTURE §9).

Run (dev setup check, owner-approved):
  uv run python -m eval.runners.judge --baseline-run <id> --agent-run <id> --setup-check 10 \
      --max-cost 1.50 --approval "<the owner's approval, word for word>"
Run (test): the same without --setup-check, on the two completed test runs.

Both source runs must be complete and cover identical question IDs (`require_same_questions`). For
each question, one Gemini call judges both systems' claims and limitations, shown as SYSTEM_A /
SYSTEM_B by the rubric's seeded coin; the A/B key goes to blinding_key.json, never to the judge.
Each item carries the full text of only the passages it cites, from the frozen corpus.

Model checks (ADR-035, methodology contingency): Google's model metadata version is checked before
the run, after every VERSION_CHECK_EVERY judged questions, and at the end; every response's
reported modelVersion is checked by `judge_client.generate`. A mismatch stops the run.

Failures (rubric v2 §5): API errors are retried with backoff (MAX_API_TRIES per call); a response
that is not valid JSON, breaks the schema or misses an item is re-asked once unchanged; afterwards
an item with exactly one valid judgment uses it, others are `not_judged`. A run that stops early
(cost cap, MAX_CONSECUTIVE_FAILURES questions failing in a row, model change, crash, Ctrl-C) is
void: VOID.json, judgments renamed, no summary (ADR-034). Every call's input, output and thinking
tokens, cost and latency are logged separately per question in calls.jsonl.
"""

import argparse
import hashlib
import html
import json
import math
import random
import re
import statistics
import sys
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from backend.app.config import REPO_ROOT
from eval import judge_client as jc
from eval import judge_rubric as jr
from eval.runners.retrieval import RESULTS_DIR, git_state
from eval.runners.run_system import (
    FINAL_SPLITS,
    check_approval,
    require_complete_run,
    require_same_questions,
)

CORPUS = REPO_ROOT / "data" / "corpus.jsonl"
VERSION_CHECK_EVERY = 25
MAX_API_TRIES = 4
BACKOFF_S = (5, 20, 60)  # or longer when Google asks (per-minute rate limit)
MAX_WAIT_S = 300
QUOTA_MARGIN = 0.10  # extra daily requests kept free for re-asks and retries
PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini daily quotas reset at midnight Pacific
MAX_CONSECUTIVE_FAILURES = 5
MAX_OUTPUT_TOKENS = 16000  # includes thinking (Google); bounds one call at ~$0.19 + input
# Per judged question, measured by the dev setup check judge-dev-20261025-192833-c72234e
# ($10.68 over ~194 test questions incl. 10% retries). Used for the approval guard only.
EST_COST_PER_QUESTION = 0.055
PER_QUESTION_ALLOWANCE = 0.25  # USD margin for the soft cost stop (one large question + retry)
VOID_FILE = "VOID.json"
VOID_JUDGMENTS = "judgments.VOID-do-not-score.jsonl"
MIN_APPROVAL_CHARS = 10
SYSTEMS = ("baseline", "agent")
_TAG = re.compile(r"<\s*(/?)\s*(question|answer_set|item|statement|passage)\b", re.IGNORECASE)

sleep = time.sleep  # replaced in tests
now = time.perf_counter


def baseline_is_a(pmid: str) -> bool:
    """Rubric v2 §3: the baseline is SYSTEM_A when this seeded draw is below 0.5."""
    return random.Random(f"judge_v1|20261024|{pmid}").random() < 0.5


def defuse(text: str) -> str:
    """Untrusted text cannot open or close the judge's own elements. Text without such tags is
    passed through byte-for-byte (as the systems saw it); otherwise it is normalized and defused."""
    folded = unicodedata.normalize("NFKC", html.unescape(text))
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    if not _TAG.search(folded):
        return text
    return _TAG.sub(lambda m: f"[{m.group(1)}{m.group(2)}", folded)


def load_corpus() -> dict[str, list[dict]]:
    with CORPUS.open(encoding="utf-8") as f:
        return {r["pmid"]: r["sections"] for r in map(json.loads, f)}


def render_passage(sections: list[dict]) -> str:
    return "\n".join(f"{defuse(s['label'])}: {defuse(s['text'])}" for s in sections)


def items_of(row: dict) -> list[dict]:
    return [{"item_id": c["id"], "kind": kind, "text": c["text"], "citation_ids": c["citation_ids"]}
            for kind, key in (("claim", "claims"), ("limitation", "limitations"))
            for c in row[key]]  # fmt: skip


def build_question(row_by_system: dict, corpus: dict, templates: dict) -> dict | None:
    """The judge request for one question, or None when neither system shows an item."""
    pmid = row_by_system["baseline"]["pmid"]
    order = ("baseline", "agent") if baseline_is_a(pmid) else ("agent", "baseline")
    key = dict(zip(jr.SYSTEM_LABELS, order, strict=True))
    if not any(items_of(row_by_system[s]) for s in SYSTEMS):
        return None
    blocks, items = [], {}
    for label, system in key.items():
        rendered = []
        for it in items_of(row_by_system[system]):
            blind_id = f"{label[-1]}.{it['item_id']}"
            passages = "\n".join(
                templates["PASSAGE_TEMPLATE"].format(
                    citation_id=cid, passage_text=render_passage(corpus[cid.split(":", 1)[1]])
                )
                for cid in it["citation_ids"]
            )
            rendered.append(templates["ITEM_TEMPLATE"].format(
                item_id=blind_id, statement=defuse(it["text"]), passages=passages))  # fmt: skip
            items[blind_id] = {"system": system, **it}
        template = "SYSTEM_BLOCK_TEMPLATE" if rendered else "EMPTY_SYSTEM_BLOCK_TEMPLATE"
        blocks.append(templates[template].format(system_label=label, items="\n".join(rendered)))
    user = templates["USER_TEMPLATE"].format(
        question=defuse(row_by_system["baseline"]["question"]),
        item_ids=", ".join(items),
        system_blocks="\n\n".join(blocks),
    )
    return {"pmid": pmid, "user": user, "items": items, "key": key}


def usable_judgments(obj, item_ids) -> dict:
    """Item ID -> its judgment, for items with exactly one valid judgment (rubric v2 §5)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("judgments"), list):
        return {}
    valid: dict[str, list] = {}
    for j in obj["judgments"]:
        if not jr.check_response({"judgments": [j]}, [j.get("item_id")] if isinstance(j, dict)
                                 else []) and j["item_id"] in item_ids:  # fmt: skip
            valid.setdefault(j["item_id"], []).append(j)
    return {i: v[0] for i, v in valid.items() if len(v) == 1}


def call_with_backoff(body: dict, attempts: list) -> dict | None:
    """One judge request, retrying retryable API errors (network, 5xx, per-minute rate limit:
    waiting at least as long as Google asks). Appends every try to `attempts`. A JudgeStop (model
    change, daily quota used up, billing/access) is never retried: it stops the run."""
    for i in range(MAX_API_TRIES):
        start = now()
        try:
            result = jc.generate(body)
        except jc.JudgeStop:
            raise
        except jc.JudgeAPIError as exc:
            attempts.append({"ok": False, "error": str(exc)[:300],
                             "latency_ms": round((now() - start) * 1000, 1)})  # fmt: skip
            if i < len(BACKOFF_S):
                sleep(min(MAX_WAIT_S, max(BACKOFF_S[i], exc.retry_after_s or 0)))
            continue
        attempts.append({"ok": True, "latency_ms": round((now() - start) * 1000, 1), **result})
        return result
    return None


def make_body(q: dict, system_instruction: str, schema: dict) -> dict:
    return jc.build_body(system_instruction, q["user"], schema, temperature=0, seed=jr.SEED,
                         max_output_tokens=MAX_OUTPUT_TOKENS, thinking_budget=None)  # fmt: skip


def request_sha256(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def judge_question(q: dict, system_instruction: str, schema: dict) -> dict:
    body = make_body(q, system_instruction, schema)
    ids = list(q["items"])
    attempts, problems, usable = [], [], {}
    for _ in range(2):  # the request, then one unchanged re-ask (rubric v2 §5)
        result = call_with_backoff(body, attempts)
        if result is None:
            problems.append("API failed after retries")
            break
        try:
            parsed = json.loads(jc.response_text(result["response"]))
            found = jr.check_response(parsed, ids)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            parsed, found = None, [f"no parseable JSON ({type(exc).__name__})"]
        attempts[-1]["format_problems"] = found
        if parsed is not None:
            usable = usable_judgments(parsed, ids)
        if not found:
            break
        problems += found
    ok = [a for a in attempts if a["ok"]]
    usage = {k: sum(a["usage"][k] for a in ok)
             for k in ("input_tokens", "output_tokens", "thinking_tokens")}  # fmt: skip
    return {
        "pmid": q["pmid"],
        "request_sha256": request_sha256(body),
        "n_items": len(ids),
        "attempts": [_slim(a) for a in attempts],
        "api_calls": len(attempts),
        "reasked": sum(1 for a in ok) > 1,
        "malformed": bool(problems),
        "problems": problems,
        "failed": not ok,
        "usage": usage,
        "cost_usd": sum(a["cost_usd"] for a in ok),
        "latency_ms": round(sum(a["latency_ms"] for a in attempts), 1),
        "judgments": usable,
    }


def _slim(attempt: dict) -> dict:
    """Keep what the record needs; drop Google's opaque thought signatures."""
    if not attempt["ok"]:
        return attempt
    response = json.loads(json.dumps(attempt["response"]))
    for cand in response.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            part.pop("thoughtSignature", None)
    return {"ok": True, "latency_ms": attempt["latency_ms"],
            "model_version_reported": attempt["model_version_reported"],
            "usage": attempt["usage"], "cost_usd": attempt["cost_usd"],
            "finish_reason": response.get("candidates", [{}])[0].get("finishReason"),
            "format_problems": attempt.get("format_problems", []),
            "response": response}  # fmt: skip


def setup_check_sample(questions: list[dict], n: int) -> list[tuple[str, str]]:
    """Dev setup check (owner: size the cap against the tail): the 4 largest requests, the 2 with
    the most items among the rest, then a seeded random fill. Returns (pmid, reason) pairs."""
    pool = sorted(questions, key=lambda q: (-len(q["user"]), q["pmid"]))
    chosen = [(q["pmid"], f"largest request #{i}") for i, q in enumerate(pool[:4], start=1)]
    rest = [q for q in pool if q["pmid"] not in dict(chosen)]
    by_items = sorted(rest, key=lambda q: (-len(q["items"]), q["pmid"]))[:2]
    chosen += [(q["pmid"], f"most items ({len(q['items'])})") for q in by_items]
    rest = sorted(q["pmid"] for q in rest if q["pmid"] not in dict(chosen))
    fill = random.Random(f"judge-setup|{jr.SEED}").sample(rest, max(0, n - len(chosen)))
    return (chosen + [(p, "random") for p in sorted(fill)])[:n]


def write_void(out, run_id: str, reason: str, done: int, planned: int) -> None:
    """Mark a judge run void. Its checkpoint (calls.jsonl) stays; resuming it needs the owner's
    fresh typed approval (`--resume`), never happens by itself, and is impossible after a model
    change (methodology: a changed judge means re-judging everything)."""
    record = {"run_id": run_id, "void": True, "reason": reason, "questions_done": done,
              "questions_planned": planned, "resumable": not reason.startswith("model changed"),
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "rule": "Judge run stopped early: void. Judgments not read or scored. Either "
                      "re-judge in full under a new run ID, or resume this run from its checkpoint "
                      "with the owner's fresh typed approval (--resume)."}  # fmt: skip
    (out / VOID_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if (out / "judgments.jsonl").exists():
        (out / "judgments.jsonl").rename(out / VOID_JUDGMENTS)


def judge_runs_of_pair(baseline_run: str, agent_run: str) -> list[tuple[dict, str]]:
    """(run_config, status) of every full (non-setup-check) judge run of this pair; status is
    "complete", "void" or "unfinished". Reads configs and markers only, never judgments."""
    found = []
    for path in sorted(RESULTS_DIR.glob("judge-*/run_config.json")):
        c = json.loads(path.read_text(encoding="utf-8"))
        if (c.get("baseline_run"), c.get("agent_run")) != (baseline_run, agent_run) or c.get(
            "setup_check"
        ) is not None:
            continue
        d = path.parent
        status = "void" if (d / VOID_FILE).exists() else (
            "complete" if (d / "summary.json").exists() else "unfinished")  # fmt: skip
        found.append((c, status))
    return found


def completed_judge_runs(baseline_run: str, agent_run: str) -> list[str]:
    return [c["run_id"] for c, s in judge_runs_of_pair(baseline_run, agent_run) if s == "complete"]


def used_approvals(config: dict) -> set[str]:
    return {a for a in [config.get("owner_approval")]
            + [r["approval"] for r in config.get("resumes", [])] if a}  # fmt: skip


def check_fresh_approval(approval: str | None, earlier: list[dict], what: str) -> str:
    """After a void run, the owner's approval must be new text, not one already spent (ADR-034)."""
    approval = (approval or "").strip()
    if len(approval) < MIN_APPROVAL_CHARS or any(approval in used_approvals(c) for c in earlier):
        raise SystemExit(f"{what} after a void judge run needs the owner's fresh typed approval "
                         "(--approval with new text); refusing.")  # fmt: skip
    return approval


def load_pair(baseline_run: str, agent_run: str) -> tuple[str, list[str], dict]:
    dirs = {"baseline": RESULTS_DIR / baseline_run, "agent": RESULTS_DIR / agent_run}
    configs = {s: json.loads((d / "run_config.json").read_text(encoding="utf-8"))
               for s, d in dirs.items()}  # fmt: skip
    for system, config in configs.items():
        if config["system"] != system:
            raise SystemExit(f"{dirs[system].name} is a {config['system']} run, not {system}.")
    split = configs["baseline"]["split"]
    if configs["agent"]["split"] != split:
        raise SystemExit("The two runs are on different splits; refusing.")
    question_ids = require_same_questions(dirs["baseline"], dirs["agent"])
    rows = {s: {r["pmid"]: r for r in require_complete_run(d)} for s, d in dirs.items()}
    return split, question_ids, rows


def build_questions(question_ids, rows, corpus, templates) -> list[dict]:
    return [
        q
        for pmid in question_ids
        if (q := build_question({s: rows[s][pmid] for s in SYSTEMS}, corpus, templates))
    ]


def logged_requests_today(model: str) -> int:
    """Requests to `model` our own records show for today's Pacific day (Gemini's quota day):
    every attempt in judge runs started today, plus the M4-T0 test call(s). A lower bound: a run
    started before midnight that ran past it is counted on its start day."""
    today = datetime.now(UTC).astimezone(PACIFIC).date()
    total = 0
    for d in sorted(RESULTS_DIR.glob("judge-*")):
        if d.name.startswith("judge-testcall-"):
            started = datetime.strptime(d.name[15:], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
            result = json.loads((d / "result.json").read_text(encoding="utf-8"))
            if started.astimezone(PACIFIC).date() == today and result["model_requested"] == model:
                total += 2  # one countTokens + one generateContent
            continue
        config_path = d / "run_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        started = datetime.fromisoformat(config["created_utc"]).astimezone(PACIFIC)
        if config.get("judge_model") == model and started.date() == today and \
                (d / "calls.jsonl").exists():  # fmt: skip
            with (d / "calls.jsonl").open(encoding="utf-8") as f:
                total += sum(len(json.loads(line)["attempts"]) for line in f)
    return total


def check_daily_quota(n_questions: int, limit: int | None, reported_used: int | None) -> dict:
    """Refuse to start unless today's remaining requests cover every question plus a margin for
    re-asks/retries. Google has no API for remaining quota, so the owner reads the limit and
    today's usage from AI Studio; our own logged requests are used if they are higher."""
    if limit is None or reported_used is None:
        raise SystemExit("--rpd-limit and --rpd-used-today are required: read both from Google AI "
                         "Studio (Google offers no API for remaining quota).")  # fmt: skip
    logged = logged_requests_today(jc.JUDGE_MODEL)
    used = max(reported_used, logged)
    margin = math.ceil(n_questions * QUOTA_MARGIN)
    needed, remaining = n_questions + margin, limit - used
    now_pt = datetime.now(UTC).astimezone(PACIFIC)
    reset = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    record = {"rpd_limit": limit, "used_reported_by_owner": reported_used,
              "used_logged_by_us": logged, "used": used, "remaining": remaining,
              "questions": n_questions, "margin": margin, "needed": needed,
              "checked_pacific": now_pt.isoformat(timespec="minutes"),
              "resets_pacific": reset.isoformat(timespec="minutes")}  # fmt: skip
    if remaining < needed:
        raise SystemExit(
            f"Daily request limit: {remaining} left today ({limit} - {used} used; AI Studio says "
            f"{reported_used}, our logs {logged}); {needed} needed ({n_questions} questions + "
            f"{margin} for re-asks/retries). Resets {reset:%Y-%m-%d %H:%M %Z}. Refusing to start."
        )
    return record


def save_checks(out, checks: list) -> None:
    (out / "version_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")


def judge_loop(
    out, run_id, questions, templates, schema, max_cost, calls, judgments, checks, planned
):
    """Judge `questions` in order, appending each one to calls.jsonl / judgments.jsonl as soon as
    it is done (the checkpoint). `calls` / `judgments` may already hold a resumed run's questions;
    they count toward the cost cap and the version-check schedule. Returns the stop reason."""
    stop_reason, streak = None, 0
    try:
        with (out / "calls.jsonl").open("a", encoding="utf-8") as cf, \
                (out / "judgments.jsonl").open("a", encoding="utf-8") as jf:  # fmt: skip
            for q in questions:
                i = len(calls)
                if i and i % VERSION_CHECK_EVERY == 0:
                    checks.append({"at": f"after {i} questions", **jc.verify_model()})
                    save_checks(out, checks)
                if max_cost is not None:
                    spent = sum(c["cost_usd"] for c in calls)
                    margin = max([PER_QUESTION_ALLOWANCE] + [2 * c["cost_usd"] for c in calls])
                    if spent + margin > max_cost:
                        stop_reason = f"cost cap ${max_cost} before question {i + 1} "
                        stop_reason += f"(spent ${spent:.4f})"
                        break
                call = judge_question(q, templates["SYSTEM_INSTRUCTION"], schema)
                calls.append(call)
                cf.write(json.dumps(call) + "\n")
                cf.flush()
                for blind_id, it in q["items"].items():
                    j = call["judgments"].get(blind_id)
                    judgments.append({"pmid": q["pmid"], "blind_id": blind_id, **it,
                                      "label": j["label"] if j else "not_judged",
                                      "evidence_quote": j["evidence_quote"] if j else None,
                                      "reason": j["reason"] if j else None})  # fmt: skip
                    jf.write(json.dumps(judgments[-1]) + "\n")
                jf.flush()
                print(f"  {i + 1}/{planned} {q['pmid']}: {call['usage']} ${call['cost_usd']:.4f} "
                      f"{call['latency_ms'] / 1000:.1f}s"
                      + (" MALFORMED" if call["malformed"] else ""), flush=True)  # fmt: skip
                streak = streak + 1 if call["failed"] else 0
                if streak >= MAX_CONSECUTIVE_FAILURES and q is not questions[-1]:
                    stop_reason = f"{streak} questions in a row failed"
                    break
            if stop_reason is None:
                checks.append({"at": "end", **jc.verify_model()})
                save_checks(out, checks)
    except jc.JudgeModelChanged as exc:
        stop_reason = f"model changed: {exc}"
    except jc.JudgeStop as exc:  # daily quota or billing/access: void, resumable with approval
        stop_reason = f"{type(exc).__name__}: {str(exc)[:300]}"
    except BaseException as exc:
        save_checks(out, checks)
        write_void(out, run_id, f"interrupted: {type(exc).__name__}", len(calls), planned)
        raise
    save_checks(out, checks)
    return stop_reason


def finish(out, run_id, config, planned, calls, judgments, checks, stop_reason, wall_s) -> None:
    if stop_reason is None and len(calls) != planned:
        stop_reason = "fewer judged questions than planned"
    if stop_reason is not None:
        write_void(out, run_id, stop_reason, len(calls), planned)
        spent = sum(c["cost_usd"] for c in calls)
        raise SystemExit(f"VOID judge run {run_id}: {stop_reason}; ${spent:.4f} spent. Not scored. "
                         "Re-judge in full under a new run ID, or resume only with the owner's "
                         "fresh typed approval.")  # fmt: skip
    summary = {"run_id": run_id, "complete": True, "judge_model": jc.JUDGE_MODEL,
               "model_versions_reported": sorted({a["model_version_reported"] for c in calls
                                                  for a in c["attempts"] if a["ok"]}),
               "version_checks": [v["version"] for v in checks],
               "rubric_file_sha256": jr.file_hashes(),
               "resumes": config.get("resumes", []),
               **summarize(calls, judgments, wall_s)}  # fmt: skip
    summary["wall_clock_s_note"] = "this session only" if summary["resumes"] else "whole run"
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "labels"}, indent=2))
    print(f"Saved to {out.relative_to(REPO_ROOT)}")


def resume(args) -> None:
    """Continue a void judge run from its checkpoint (owner's fresh typed approval only). Refused
    if the run is not resumable (model change), or if anything the judge reads, the rubric, the
    model or the blinding would differ from the stopped run."""
    out = RESULTS_DIR / args.resume
    if not (out / VOID_FILE).exists():
        raise SystemExit(f"{args.resume} is not a stopped (void) judge run; nothing to resume.")
    void = json.loads((out / VOID_FILE).read_text(encoding="utf-8"))
    if not void.get("resumable"):
        raise SystemExit(f"{args.resume} is not resumable ({void['reason']}); re-judge in full.")
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    approval = check_fresh_approval(args.approval, [config], "Resuming")
    if args.max_cost is None:
        raise SystemExit(
            "--max-cost (the cap for the whole run, incl. what was spent) is required."
        )
    jr.check_frozen()
    pinned = (jc.JUDGE_MODEL, jc.EXPECTED_MODEL_VERSION, jc.EXPECTED_RESPONSE_MODEL_VERSION)
    recorded = (config["judge_model"], config["expected_model_version"],
                config["expected_response_model_version"])  # fmt: skip
    if (jr.file_hashes() != config["rubric_file_sha256"]
            or jr.JUDGE_VERSION != config["rubric_version"] or pinned != recorded):  # fmt: skip
        raise SystemExit("Rubric or judge model differs from the stopped run; refusing to resume.")
    _, question_ids, rows = load_pair(config["baseline_run"], config["agent_run"])
    corpus, templates, schema = load_corpus(), jr.load_prompt(), jr.load_schema()
    by_pmid = {q["pmid"]: q for q in build_questions(question_ids, rows, corpus, templates)}
    questions = [by_pmid[p] for p in config["question_ids"]]
    key = json.loads((out / "blinding_key.json").read_text(encoding="utf-8"))
    if key != {q["pmid"]: q["key"] for q in questions}:
        raise SystemExit("Blinding would differ from the stopped run; refusing to resume.")
    done = [json.loads(line) for line in (out / "calls.jsonl").open(encoding="utf-8")]
    if [c["pmid"] for c in done] != config["question_ids"][: len(done)]:
        raise SystemExit("The checkpoint is not a prefix of the planned questions; refusing.")
    for c in done:  # what the judge reads must be byte-identical to what it already judged
        body = make_body(by_pmid[c["pmid"]], templates["SYSTEM_INSTRUCTION"], schema)
        if request_sha256(body) != c["request_sha256"]:
            raise SystemExit(f"The judge request for {c['pmid']} differs from the stopped run's; "
                             "refusing to resume.")  # fmt: skip
    quota = check_daily_quota(len(questions) - len(done), args.rpd_limit, args.rpd_used_today)
    checks = json.loads((out / "version_checks.json").read_text(encoding="utf-8"))
    checks.append({"at": f"resume after {len(done)} questions", **jc.verify_model()})
    (out / VOID_JUDGMENTS).rename(out / "judgments.jsonl")
    judgments = [json.loads(line) for line in (out / "judgments.jsonl").open(encoding="utf-8")]
    config.setdefault("resumes", []).append({
        "resumed_utc": datetime.now(UTC).isoformat(timespec="seconds"), "approval": approval,
        "questions_done": len(done), "max_cost_usd": args.max_cost, "void_record": void,
        "daily_quota_check": quota,
        **git_state()})  # fmt: skip
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / VOID_FILE).unlink()
    print(f"Resuming {args.resume}: {len(done)} done, {len(questions) - len(done)} to judge.")
    wall_start = now()
    stop = judge_loop(out, args.resume, questions[len(done):], templates, schema, args.max_cost,
                      done, judgments, checks, len(questions))  # fmt: skip
    finish(out, args.resume, config, len(questions), done, judgments, checks, stop,
           now() - wall_start)  # fmt: skip


def pct(values: list[float], q: int):
    if not values:
        return None
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def describe(values: list[float]) -> dict:
    return {"median": pct(values, 50), "p95": pct(values, 95), "max": max(values, default=None),
            "mean": statistics.mean(values) if values else None}  # fmt: skip


def summarize(calls: list[dict], judgments: list[dict], wall_s: float) -> dict:
    labels: dict = {}
    for j in judgments:
        bucket = labels.setdefault(j["system"], {}).setdefault(j["kind"], {})
        bucket[j["label"]] = bucket.get(j["label"], 0) + 1
    return {
        "questions_judged": len(calls),
        "items": len(judgments),
        "not_judged": sum(j["label"] == "not_judged" for j in judgments),
        "labels": labels,
        "malformed_questions": [c["pmid"] for c in calls if c["malformed"]],
        "reasked_questions": [c["pmid"] for c in calls if c["reasked"]],
        "api_retry_questions": [c["pmid"] for c in calls
                                if any(not a["ok"] for a in c["attempts"])],  # fmt: skip
        "failed_questions": [c["pmid"] for c in calls if c["failed"]],
        "tokens_per_question": {k: describe([c["usage"][k] for c in calls])
                                for k in ("input_tokens", "output_tokens", "thinking_tokens")},
        "cost_per_question_usd": describe([c["cost_usd"] for c in calls]),
        "latency_per_question_s": describe([c["latency_ms"] / 1000 for c in calls]),
        "cost_usd_total": round(sum(c["cost_usd"] for c in calls), 6),
        "wall_clock_s": round(wall_s, 1),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run")
    parser.add_argument("--agent-run")
    parser.add_argument("--setup-check", type=int, help="dev only: judge this many questions")
    parser.add_argument("--max-cost", type=float, help="soft spending stop (USD)")
    parser.add_argument("--approval", help="the owner's approval text, word for word (saved)")
    parser.add_argument("--rerun-reason", help="test only: why a judged pair is judged again")
    parser.add_argument("--resume", help="a void judge run to continue (owner's fresh approval)")
    parser.add_argument(
        "--rpd-limit", type=int, help="the model's requests-per-day limit (AI Studio)"
    )
    parser.add_argument("--rpd-used-today", type=int, help="requests used today, per AI Studio")
    args = parser.parse_args(argv)
    if args.resume:
        if args.baseline_run or args.agent_run or args.setup_check is not None:
            raise SystemExit("--resume takes the stopped run's own settings; give only --resume, "
                             "--max-cost and --approval.")  # fmt: skip
        return resume(args)
    if not (args.baseline_run and args.agent_run):
        raise SystemExit("--baseline-run and --agent-run are required (or --resume).")

    jr.check_frozen()
    split, question_ids, rows = load_pair(args.baseline_run, args.agent_run)
    if args.setup_check is not None and split != "dev":
        raise SystemExit(
            "--setup-check is for dev runs only; a frozen test split is judged in full."
        )
    earlier_void = []
    if args.setup_check is None:
        runs = judge_runs_of_pair(args.baseline_run, args.agent_run)
        earlier = [c["run_id"] for c, s in runs if s == "complete"]
        if split in FINAL_SPLITS and earlier and (args.rerun_reason is None
                                            or len(args.rerun_reason.strip()) < 20):  # fmt: skip
            raise SystemExit(f"{split} pair already judged ({', '.join(earlier)}); a re-judge "
                             "needs --rerun-reason (>= 20 chars), saved with the run.")  # fmt: skip
        earlier_void = [c for c, s in runs if s in ("void", "unfinished")]
        if earlier_void:
            check_fresh_approval(args.approval, earlier_void, "A re-judge")

    corpus, templates, schema = load_corpus(), jr.load_prompt(), jr.load_schema()
    questions = build_questions(question_ids, rows, corpus, templates)
    selection = None
    if args.setup_check is not None:
        selection = setup_check_sample(questions, args.setup_check)
        wanted = dict(selection)
        questions = [q for q in questions if q["pmid"] in wanted]
    estimate = round(len(questions) * EST_COST_PER_QUESTION, 4)
    check_approval(estimate, args.max_cost, args.approval)  # before any folder or paid call
    quota = check_daily_quota(len(questions), args.rpd_limit, args.rpd_used_today)

    checks = [{"at": "start", **jc.verify_model()}]  # free; refuses a changed model
    git = git_state()
    run_id = f"judge-{split}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{git['git_commit'][:7]}"
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=False)
    save_checks(out, checks)
    config = {
        "run_id": run_id, "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), **git,
        "split": split, "baseline_run": args.baseline_run, "agent_run": args.agent_run,
        "setup_check": selection, "question_ids": [q["pmid"] for q in questions],
        "questions_without_items": [p for p in question_ids
                                    if p not in {q["pmid"] for q in questions}] if not selection
        else None,
        "earlier_void_judge_runs": [c["run_id"] for c in earlier_void],
        "judge_provider": jc.JUDGE_PROVIDER, "judge_model": jc.JUDGE_MODEL,
        "expected_model_version": jc.EXPECTED_MODEL_VERSION,
        "expected_response_model_version": jc.EXPECTED_RESPONSE_MODEL_VERSION,
        "judge_price_per_m_tokens": jc.JUDGE_PRICE_PER_M_TOKENS,
        "rubric_version": jr.JUDGE_VERSION, "rubric_file_sha256": jr.file_hashes(),
        "generation": {"temperature": 0, "seed": jr.SEED, "max_output_tokens": MAX_OUTPUT_TOKENS,
                       "thinking": "model default", "response_schema": "judge_v1_schema.json"},
        "blinding": "baseline is SYSTEM_A when "
                    'random.Random(f"judge_v1|20261024|{pmid}").random() < 0.5',
        "version_check_every": VERSION_CHECK_EVERY, "estimated_cost_usd": estimate,
        "max_cost_usd": args.max_cost,
        "owner_approval": args.approval.strip() if args.approval else None,
        "rerun_reason": args.rerun_reason, "daily_quota_check": quota,
    }  # fmt: skip
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    key = {q["pmid"]: q["key"] for q in questions}
    (out / "blinding_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    wall_start = now()
    calls, judgments = [], []
    stop = judge_loop(out, run_id, questions, templates, schema, args.max_cost, calls, judgments,
                      checks, len(questions))  # fmt: skip
    finish(out, run_id, config, len(questions), calls, judgments, checks, stop, now() - wall_start)


if __name__ == "__main__":
    main(sys.argv[1:])
