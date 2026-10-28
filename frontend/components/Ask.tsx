"use client";

import { useEffect, useRef, useState } from "react";
import Becky from "./Becky";
import { AskError, askStream, type Claim, type QueryResponse, type StageEvent } from "@/lib/stream";

// Set at build time; next.config.ts refuses a production build without it.
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const MIN_CHARS = 5;
const MAX_CHARS = 500;

type State =
  | { kind: "idle" }
  | { kind: "working"; stage: StageEvent | null }
  | { kind: "done"; result: QueryResponse }
  | { kind: "failed"; message: string };

const VERDICT_LABEL: Record<string, string> = { yes: "Yes", no: "No", maybe: "Maybe" };

// Friendly, fixed wording per abstain reason (backend/app/agents/validator.py), never model prose.
const NO_ANSWER_TEXT: Record<string, string> = {
  weak_evidence: "The abstracts Becky found aren't close enough to this question, so she won't guess.",
  no_supported_claims: "Becky couldn't back up an answer with the abstracts she found, so she won't guess.",
  model_abstained: "The abstracts Becky found don't answer this question directly, so she won't guess.",
};
const NO_ANSWER_FALLBACK = "The abstracts Becky found don't answer this clearly, so she won't guess.";

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });
  const controller = useRef<AbortController | null>(null);
  const results = useRef<HTMLDivElement>(null);
  const answerHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => () => controller.current?.abort(), []);

  // Bring Becky's progress into view when a question starts (it is below the fold on phones).
  const startedWorking = state.kind === "working" && state.stage === null;
  useEffect(() => {
    if (!startedWorking) return;
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    results.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "nearest" });
  }, [startedWorking]);

  // When the answer arrives, move focus to it so screen readers start reading there.
  const finished = state.kind === "done";
  useEffect(() => {
    if (finished) answerHeading.current?.focus();
  }, [finished]);

  const trimmed = question.trim();
  const tooShort = trimmed.length < MIN_CHARS;
  const working = state.kind === "working";

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (tooShort || working) return;
    controller.current?.abort();
    const ctrl = new AbortController();
    controller.current = ctrl;
    setState({ kind: "working", stage: null });
    try {
      const onStage = (stage: StageEvent) => {
        if (stage.message) setState({ kind: "working", stage }); // keep the last message if one is empty
      };
      const result = await askStream(API_BASE, trimmed, onStage, ctrl.signal);
      setState({ kind: "done", result });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setState({ kind: "failed", message: err instanceof AskError ? err.message : "Something went wrong while answering, please try again later." });
    }
  }

  return (
    <>
      <section className="hero">
        <div className="becky-col">
          <div className="bubble">Hi, I&apos;m Becky! I read research abstracts and link every finding to the study it&apos;s based on.</div>
          <Becky pose="normal" />
        </div>
        <form onSubmit={submit}>
          <h1>Ask a research question</h1>
          <p className="lede">
            GroundedMed answers from a fixed set of medical research abstracts, and every finding links to the study
            it&apos;s based on, so you can check it yourself.
          </p>
          <label htmlFor="q" className="field-label">Your question</label>
          <textarea
            id="q"
            value={question}
            maxLength={MAX_CHARS}
            aria-describedby="q-hint"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="e.g. Does keyhole surgery hurt newborns less than open surgery?"
          />
          <div className="ask-row">
            <span className="char-count" id="q-hint">
              {tooShort ? `At least ${MIN_CHARS} characters · ` : ""}
              {question.length}/{MAX_CHARS}
            </span>
            <button className="btn" type="submit" disabled={tooShort || working} aria-describedby="q-hint">
              {working ? "Becky is reading…" : "Ask Becky"}
            </button>
          </div>
        </form>
      </section>

      <div ref={results} className="results">
        {state.kind === "working" && (
          <section className="result working">
            <Becky pose="working" width={140} />
            <div>
              <h2>Becky is reading the abstracts…</h2>
              <p className="stage-now" role="status">
                {state.stage?.message || "Starting"}
              </p>
            </div>
          </section>
        )}
        {state.kind === "failed" && (
          <section className="result">
            <div className="error-box" role="alert">
              {state.message}
            </div>
          </section>
        )}
        {state.kind === "done" && <Answer result={state.result} headingRef={answerHeading} />}
      </div>
    </>
  );
}

function Answer({ result, headingRef }: { result: QueryResponse; headingRef: React.Ref<HTMLHeadingElement> }) {
  // Number sources in order of first citation, so [1] is the first source the answer uses.
  const order: string[] = [];
  for (const item of [...result.claims, ...result.limitations]) {
    for (const id of item.citation_ids) if (!order.includes(id)) order.push(id);
  }
  const byId = new Map(result.citations.map((c) => [c.id, c]));
  const list = (items: Claim[]) => (
    <ul className="findings">
      {items.map((c) => (
        <li key={c.id}>
          {c.text}
          <span className="cite"> {c.citation_ids.map((id) => `[${order.indexOf(id) + 1}]`).join("")}</span>
        </li>
      ))}
    </ul>
  );
  const findings = (
    <>
      {result.claims.length > 0 && list(result.claims)}
      {result.limitations.length > 0 && (
        <>
          <h3 className="limits-title">Limitations</h3>
          {list(result.limitations)}
        </>
      )}
      {order.length > 0 && (
        <ol className="sources" aria-label="Sources">
          {order.map((id, i) => {
            const c = byId.get(id);
            return (
              <li key={id}>
                [{i + 1}] PMID {c?.pmid ?? id.replace("PMID:", "")}
                {c && (
                  <>
                    {" · "}
                    <a href={c.pubmed_url} target="_blank" rel="noopener noreferrer">
                      PubMed
                    </a>
                  </>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </>
  );

  if (result.abstained) {
    return (
      <section className="result">
        <div className="no-answer">
          <Becky pose="noAnswer" width={140} />
          <div className="refusal">
            <h2 ref={headingRef} tabIndex={-1} className="refusal-title">
              Not enough evidence
            </h2>
            {NO_ANSWER_TEXT[result.abstain_reason ?? ""] ?? NO_ANSWER_FALLBACK}
          </div>
        </div>
        {result.claims.length + result.limitations.length > 0 && (
          <>
            <h3 className="limits-title">What the abstracts do say</h3>
            {findings}
          </>
        )}
      </section>
    );
  }
  return (
    <section className="result">
      <h2 ref={headingRef} tabIndex={-1}>
        Becky&apos;s answer
      </h2>
      <div className="verdict">{VERDICT_LABEL[result.verdict] ?? result.verdict}</div>
      {findings}
    </section>
  );
}
