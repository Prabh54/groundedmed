import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import type { StageEvent } from "./stream.ts";
import {
  DID_NOT_FINISH, NOT_STARTED, SKIPPED, STEP_ICON, STEP_LABELS, applyStage, doneMeta, failSteps, finishSteps,
  hasProgress, initialSteps, stepViews, type Step,
} from "./timeline.ts";

const ev = (name: string, status: StageEvent["status"], message = ""): StageEvent => ({ name, status, message, ms: 1 });
const run = (events: StageEvent[], steps: Step[] = initialSteps()) => events.reduce(applyStage, steps);
const statuses = (steps: Step[]) => steps.map((s) => s.status);
const metas = (steps: Step[]) => steps.map((s) => s.meta);

// Same wording as backend/app/agents/pipeline.py (tied to it by the last test).
const PLANNER = [ev("planner", "started", "Planning evidence needs"), ev("planner", "done", "Planned 3 search queries")];
const RETRIEVER = [
  ev("retriever", "started", "Searching the corpus"),
  ev("retriever", "done", "Retrieved 36 candidate passages; selected 5 evidence passages"),
];
const GATE_OK = [ev("gate", "started", "Checking evidence relevance"), ev("gate", "done", "Evidence passed the relevance threshold")];
const SYNTH = [ev("synthesizer", "started", "Writing a cited answer"), ev("synthesizer", "done", "Drafted 2 claims and 1 limitations")];
const CRITIC = [ev("critic", "started", "Checking 3 claims and limitations"), ev("critic", "done", "Checked 3 claims and limitations")];
const VALIDATOR = [ev("validator", "started", "Validating citations"), ev("validator", "done", "Kept 2 cited claims; removed 1 unsupported items")];
const VALIDATOR_AFTER_ERROR = [ev("validator", "started", "Validating citations"), ev("validator", "done", "No answer: a processing step failed")];
const SUCCESS = [...PLANNER, ...RETRIEVER, ...GATE_OK, ...SYNTH, ...CRITIC, ...VALIDATOR];

test("a full answer completes all four steps with counts taken from the server messages", () => {
  const steps = run(SUCCESS);
  assert.deepEqual(steps.map((s) => s.label), [...STEP_LABELS]);
  assert.deepEqual(statuses(steps), ["done", "done", "done", "done"]);
  assert.deepEqual(metas(steps), [
    "3 searches planned",
    "36 passages found, best 5 kept",
    "2 findings, 1 limitation drafted",
    "2 findings kept, 1 removed",
  ]);
  assert.deepEqual(steps.map((s) => STEP_ICON[s.status]), ["✓", "✓", "✓", "✓"]);
});

test("counts change with the server's numbers (nothing hard-coded)", () => {
  const steps = run([
    ev("planner", "done", "Planned 1 search queries"),
    ev("retriever", "done", "Retrieved 20 candidate passages; selected 4 evidence passages"),
    ev("synthesizer", "done", "Drafted 1 claims and 0 limitations"),
    ev("validator", "done", "Kept 7 cited claims; removed 0 unsupported items"),
  ]);
  assert.deepEqual(metas(steps), ["1 search planned", "20 passages found, best 4 kept", "1 finding, 0 limitations drafted", "7 findings kept, 0 removed"]);
});

test("mid-run: earlier steps ticked, the current one running with its server message, later ones waiting", () => {
  const steps = run([...PLANNER, ...RETRIEVER, ...GATE_OK, SYNTH[0]]);
  assert.deepEqual(statuses(steps), ["done", "done", "running", "todo"]);
  assert.equal(steps[2].meta, "writing a cited answer…");
  assert.equal(steps[3].meta, "");
});

test("Retrieving stays running through the relevance check and keeps its counts", () => {
  const steps = run([...PLANNER, ...RETRIEVER, GATE_OK[0]]);
  assert.equal(steps[1].status, "running");
  assert.equal(steps[1].meta, "36 passages found, best 5 kept");
  const checking = run([...PLANNER, ...RETRIEVER, ...GATE_OK, ...SYNTH, ...CRITIC]);
  assert.equal(checking[3].status, "running", "critic done is not the end of Checking claims");
});

test("weak evidence: writing and checking are skipped; the server's validator events do not undo that", () => {
  const weak = [
    ...PLANNER, ...RETRIEVER,
    ev("gate", "started", "Checking evidence relevance"),
    ev("gate", "done", "Evidence scored below the relevance threshold"),
    ev("validator", "started", "Validating citations"),
    ev("validator", "done", "No answer: the evidence scored below the relevance threshold"),
  ];
  const steps = finishSteps(run(weak));
  assert.deepEqual(statuses(steps), ["done", "done", "skipped", "skipped"]);
  assert.deepEqual(metas(steps), ["3 searches planned", "nothing close enough", SKIPPED, SKIPPED]);
  assert.deepEqual(steps.map((s) => STEP_ICON[s.status]), ["✓", "✓", "–", "–"]);
});

// The real failure order per step (backend tests: test_stream_provider_failure_ends_with_an_error_event):
// the node reports `failed`; after an AI-service failure the graph still runs the validator; then an error event.
const FAILURES: [string, number, StageEvent[]][] = [
  ["planner", 0, [PLANNER[0], ev("planner", "failed", "This step failed"), ...VALIDATOR_AFTER_ERROR]],
  ["retriever", 1, [...PLANNER, RETRIEVER[0], ev("retriever", "failed", "This step failed")]],
  ["synthesizer", 2, [...PLANNER, ...RETRIEVER, ...GATE_OK, SYNTH[0], ev("synthesizer", "failed", "This step failed"), ...VALIDATOR_AFTER_ERROR]],
  ["critic", 3, [...PLANNER, ...RETRIEVER, ...GATE_OK, ...SYNTH, CRITIC[0], ev("critic", "failed", "This step failed"), ...VALIDATOR_AFTER_ERROR]],
];

for (const [node, at, events] of FAILURES) {
  test(`${node} failure: that step shows the red cross, later steps say "not started", earlier stay ticked`, () => {
    const steps = failSteps(run(events)); // the error event arrives last
    steps.forEach((s, i) => {
      if (i < at) assert.deepEqual([s.status, STEP_ICON[s.status]], ["done", "✓"], `${s.label} before the failure`);
      if (i === at) assert.deepEqual([s.status, STEP_ICON[s.status], s.meta], ["failed", "✕", DID_NOT_FINISH], s.label);
      if (i > at) assert.deepEqual([s.status, s.meta], ["todo", NOT_STARTED], `${s.label} after the failure`);
    });
    assert.equal(steps.filter((s) => s.status === "failed").length, 1);
  });
}

test("an error with no failed event (connection cut, server bug) marks the step that was running", () => {
  const steps = failSteps(run([...PLANNER, ...RETRIEVER, ...GATE_OK, SYNTH[0]]));
  assert.deepEqual(statuses(steps), ["done", "done", "failed", "todo"]);
  assert.deepEqual(metas(steps).slice(2), [DID_NOT_FINISH, NOT_STARTED]);
  const betweenSteps = failSteps(run(PLANNER));
  assert.deepEqual(statuses(betweenSteps), ["done", "failed", "todo", "todo"]);
});

test("an error before any server event marks no step and shows no timeline (429, API down)", () => {
  assert.deepEqual(failSteps(initialSteps()), initialSteps());
  assert.equal(hasProgress(failSteps(initialSteps())), false);
  assert.equal(hasProgress(run(PLANNER.slice(0, 1))), true);
});

test("events after the result is final change nothing, and finishing ticks a still-running step", () => {
  const steps = finishSteps(run(SUCCESS.slice(0, -1))); // validator done never seen
  assert.deepEqual(statuses(steps), ["done", "done", "done", "done"]);
  assert.equal(run([ev("synthesizer", "started", "x")], failSteps(run(PLANNER)))[2].status, "todo");
  assert.deepEqual(run([ev("unknown_node", "started", "x")]), initialSteps());
});

test("an unknown message format shows the server's own words", () => {
  assert.equal(doneMeta("planner", "Made a plan"), "Made a plan");
  assert.equal(doneMeta("validator", "No answer: no claim was supported by its cited evidence"), "no claim was supported by its cited evidence");
});

test("the parsed message formats still exist in the backend (pipeline.py)", () => {
  const src = readFileSync(new URL("../../backend/app/agents/pipeline.py", import.meta.url), "utf8");
  for (const fragment of [
    `f"Planned {len(update['subquestions'])} search queries"`,
    `f"Retrieved {len(update['retrieved_candidates'])} candidate passages; "`,
    `f"selected {len(update['selected_evidence'])} evidence passages{note}"`,
    `"Evidence scored below the relevance threshold"`,
    `f"Drafted {len(draft.claims)} claims and {len(draft.limitations)} limitations"`,
    `f"Checked {_items(state)} claims and limitations"`,
    `f"Kept {len(final.claims)} cited claims; removed {removed} unsupported items"`,
    `"No answer: a processing step failed"`,
    `"planner": "Planning evidence needs"`,
    `"synthesizer": "Writing a cited answer"`,
  ]) {
    assert.ok(src.includes(fragment), `pipeline.py no longer has: ${fragment}`);
  }
});

// What the page draws (Timeline.tsx renders stepViews as-is: dot = icon, text = label · meta, sr-only = srText).
const drawn = (steps: Step[], variant: "list" | "trail") =>
  stepViews(steps, variant).map((v) => `${v.icon || "·"} ${v.label}${v.meta ? ` · ${v.meta}` : ""}${v.srText ? ` (${v.srText})` : ""}`);

for (const [node, at, events] of FAILURES) {
  test(`${node} failure as drawn: red cross on that step, "not started" on every later step`, () => {
    const lines = drawn(failSteps(run(events)), "list");
    lines.forEach((line, i) => {
      if (i < at) assert.match(line, /^✓ /, line);
      if (i === at) assert.equal(line, `✕ ${STEP_LABELS[i]} · didn't finish (failed)`);
      if (i > at) assert.equal(line, `· ${STEP_LABELS[i]} · not started`, "no contradictory 'waiting' either");
    });
  });
}

test("as drawn: the slim row above an answer shows only the last step's counts; skipped steps say why", () => {
  assert.deepEqual(drawn(finishSteps(run(SUCCESS)), "trail"), [
    "✓ Planning (done)", "✓ Retrieving (done)", "✓ Synthesizing (done)", "✓ Checking claims · 2 findings kept, 1 removed (done)",
  ]);
  const weak = [...PLANNER, ...RETRIEVER, ev("gate", "done", "Evidence scored below the relevance threshold")];
  assert.deepEqual(drawn(finishSteps(run(weak)), "trail"), [
    "✓ Planning (done)", "✓ Retrieving · nothing close enough (done)", "– Synthesizing · skipped", "– Checking claims · skipped",
  ]);
  const views = stepViews(run([...PLANNER, RETRIEVER[0]]), "list");
  assert.deepEqual(views.map((v) => v.current), [false, true, false, false], "aria-current on the running step only");
});

test("reranker unavailable: the counts still show", () => {
  const steps = run([ev("retriever", "done", "Retrieved 20 candidate passages; selected 5 evidence passages (reranker unavailable: similarity order)")]);
  assert.equal(steps[1].meta, "20 passages found, best 5 kept");
});

test("a failure in the relevance check marks Retrieving", () => {
  const steps = failSteps(run([...PLANNER, ...RETRIEVER, GATE_OK[0], ev("gate", "failed", "This step failed")]));
  assert.deepEqual(statuses(steps), ["done", "failed", "todo", "todo"]);
});

test("model abstained / no supported claims: all steps ticked, the server's reason shown on Checking claims", () => {
  for (const [message, meta] of [
    ["No answer: the evidence does not address the question", "the evidence does not address the question"],
    ["No answer: no claim was supported by its cited evidence", "no claim was supported by its cited evidence"],
  ]) {
    const steps = finishSteps(run([...SUCCESS.slice(0, -1), ev("validator", "done", message)]));
    assert.deepEqual(statuses(steps), ["done", "done", "done", "done"]);
    assert.equal(steps[3].meta, meta);
  }
});

test("an error after every step finished (answer lost on the way) leaves the ticks: those steps did finish", () => {
  const steps = failSteps(run(SUCCESS));
  assert.deepEqual(statuses(steps), ["done", "done", "done", "done"]);
});
