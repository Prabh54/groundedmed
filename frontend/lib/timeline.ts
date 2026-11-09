// The four visible steps built from /query/stream stage events. The server runs six nodes
// (backend/app/agents/pipeline.py): Retrieving = retriever + gate, Checking claims = critic + validator.
// Every count shown comes from a server message; nothing here invents progress or shows reasoning.
import type { StageEvent } from "./stream";

export type StepStatus = "todo" | "running" | "done" | "skipped" | "failed";
export type Step = { label: string; status: StepStatus; meta: string };

export const STEP_LABELS = ["Planning", "Retrieving", "Synthesizing", "Checking claims"] as const;

const NODE_STEP: Record<string, number> = { planner: 0, retriever: 1, gate: 1, synthesizer: 2, critic: 3, validator: 3 };
// The node whose `done` completes its step; an earlier node's `done` only updates the counts.
const LAST_NODE = new Set(["planner", "gate", "synthesizer", "validator"]);
const WEAK_GATE = "below the relevance threshold"; // gate `done` message when the evidence is too weak

// Dot content per status (the component draws these; mint tick, grey dash, red cross).
export const STEP_ICON: Record<StepStatus, string> = { todo: "", running: "", done: "✓", skipped: "–", failed: "✕" };

export const NOT_STARTED = "not started";
export const DID_NOT_FINISH = "didn't finish";
export const SKIPPED = "skipped";

export function initialSteps(): Step[] {
  return STEP_LABELS.map((label) => ({ label, status: "todo", meta: "" }));
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** Short friendly text for a `done` message; the server's own words if the format is unknown. */
export function doneMeta(name: string, message: string): string {
  let m: RegExpMatchArray | null;
  if (name === "planner" && (m = message.match(/^Planned (\d+) search quer/))) {
    return `${plural(+m[1], "search", "searches")} planned`;
  }
  if (name === "retriever" && (m = message.match(/^Retrieved (\d+) candidate passages?; selected (\d+) evidence passages?/))) {
    return `${plural(+m[1], "passage", "passages")} found, best ${m[2]} kept`;
  }
  if (name === "gate") return message.includes(WEAK_GATE) ? "nothing close enough" : ""; // passed: keep the counts
  if (name === "synthesizer" && (m = message.match(/^Drafted (\d+) claims? and (\d+) limitations?/))) {
    return `${plural(+m[1], "finding", "findings")}, ${plural(+m[2], "limitation", "limitations")} drafted`;
  }
  if (name === "critic" && (m = message.match(/^Checked (\d+) claims and limitations/))) {
    return `${plural(+m[1], "item", "items")} checked`;
  }
  if (name === "validator" && (m = message.match(/^Kept (\d+) cited claims?; removed (\d+) unsupported items?/))) {
    return `${plural(+m[1], "finding", "findings")} kept, ${m[2]} removed`;
  }
  if (name === "validator" && message.startsWith("No answer: ")) return message.slice("No answer: ".length);
  return message;
}

const runningMeta = (message: string) => (message ? `${message[0].toLowerCase()}${message.slice(1)}…` : "");

const halted = (steps: Step[]) => steps.some((s) => s.status === "failed" || s.status === "skipped");

/** Applies one stage event. After a failure or a weak-evidence gate the timeline is final:
 *  the server still runs its validator then, but that is bookkeeping, not a visible step. */
export function applyStage(steps: Step[], event: StageEvent): Step[] {
  const i = NODE_STEP[event.name];
  if (i === undefined || halted(steps)) return steps;
  const next = steps.map((s) => ({ ...s }));
  const step = next[i];
  if (event.status === "started") {
    step.status = "running";
    if (event.name !== "gate") step.meta = runningMeta(event.message); // gate keeps the found/kept counts
  } else if (event.status === "done") {
    const meta = doneMeta(event.name, event.message);
    if (meta) step.meta = meta;
    if (LAST_NODE.has(event.name)) step.status = "done";
    if (event.name === "gate" && event.message.includes(WEAK_GATE)) {
      for (const later of next.slice(i + 1)) Object.assign(later, { status: "skipped", meta: SKIPPED });
    }
  } else if (event.status === "failed") {
    return failAt(next, i);
  }
  return next;
}

function failAt(steps: Step[], i: number): Step[] {
  return steps.map((s, j) =>
    j === i ? { ...s, status: "failed", meta: DID_NOT_FINISH } : j > i && s.status === "todo" ? { ...s, meta: NOT_STARTED } : s,
  );
}

/** The request ended in an error: if no step reported `failed`, the unfinished step is the one that failed.
 *  Nothing is marked when the server never started (429, 422, API down): no step ran, so none failed. */
export function failSteps(steps: Step[]): Step[] {
  if (halted(steps) || !hasProgress(steps)) return steps;
  const i = steps.findIndex((s) => s.status === "running" || s.status === "todo");
  return i === -1 ? steps : failAt(steps, i);
}

/** The answer arrived: a step still running is finished, one that never ran was skipped. */
export function finishSteps(steps: Step[]): Step[] {
  return steps.map((s) =>
    s.status === "running" ? { ...s, status: "done" } : s.status === "todo" ? { ...s, status: "skipped", meta: SKIPPED } : s,
  );
}

/** True once the server has reported anything, so a timeline is worth showing next to an error. */
export const hasProgress = (steps: Step[]) => steps.some((s) => s.status !== "todo");

// Screen-reader status per step (the dot itself is decorative); left out when the visible text says it.
const STATUS_TEXT: Record<StepStatus, string> = { todo: "waiting", running: "in progress", done: "done", skipped: "", failed: "failed" };

export type StepView = { label: string; status: StepStatus; icon: string; meta: string; srText: string; current: boolean };

/** What each step draws. "list": every step's text. "trail" (slim row above an answer): only the counts
 *  of the last finished step and the reason a step was skipped. */
export function stepViews(steps: Step[], variant: "list" | "trail"): StepView[] {
  const lastDone = steps.map((s) => s.status).lastIndexOf("done");
  return steps.map((s, i) => {
    const meta = variant === "list" || i === lastDone || s.status === "skipped" ? s.meta : "";
    const srText = s.status === "todo" && meta ? "" : STATUS_TEXT[s.status];
    return { label: s.label, status: s.status, icon: STEP_ICON[s.status], meta, srText, current: s.status === "running" };
  });
}
