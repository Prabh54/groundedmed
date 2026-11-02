"use client";

import { useEffect, useRef, useState } from "react";
import Becky from "./Becky";
import Sources from "./Sources";
import Timeline from "./Timeline";
import { BUTTON_COUNT, CORPUS_ABSTRACTS, EXAMPLES, ROTATE_MS, nextExample } from "@/lib/examples";
import { AskError, askStream, type Claim, type QueryResponse, type StageEvent } from "@/lib/stream";
import { anchorId, countText, sourceViews } from "@/lib/sources";
import { applyStage, failSteps, finishSteps, hasProgress, initialSteps, type Step } from "@/lib/timeline";

// Set at build time; next.config.ts refuses a production build without it.
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const MIN_CHARS = 5;
const MAX_CHARS = 500;

type State =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "done"; result: QueryResponse }
  | { kind: "failed"; message: string };

const VERDICT_LABEL: Record<string, string> = { yes: "Yes", no: "No", maybe: "Maybe" };

// Friendly, fixed wording per abstain reason (backend/app/agents/validator.py), never model prose.
const NO_ANSWER_TEXT: Record<string, string> = {
  weak_evidence:
    "Becky couldn't find a paper that closely matches this question, so she won't guess. Try rephrasing with medical " +
    "terms (e.g. 'right hemisphere' instead of 'right half of the brain'), or try one of the examples below.",
  no_supported_claims: "Becky couldn't back up an answer with the abstracts she found, so she won't guess.",
  model_abstained: "The abstracts Becky found don't answer this question directly, so she won't guess.",
};
const NO_ANSWER_FALLBACK = "The abstracts Becky found don't answer this clearly, so she won't guess.";
// The API's personal-advice screen (ADR-042): refused before any search, so no steps and no sources.
const ADVICE_TITLE = "Becky can't give medical advice";
const ADVICE_TEXT =
  "This looks like a question about someone's own health or treatment. Becky only summarises research, so she " +
  "won't answer it. Please ask a doctor or pharmacist.";
const ADVICE_TIP = "Ask what studies found, e.g. 'Does X work better than Y?', not what someone should do.";

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });
  const [steps, setSteps] = useState<Step[]>(initialSteps);
  const [example, setExample] = useState(0);
  const box = useRef<HTMLTextAreaElement>(null);
  const controller = useRef<AbortController | null>(null);
  const results = useRef<HTMLDivElement>(null);
  const answerHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => () => controller.current?.abort(), []);

  // Rotate the placeholder through the examples; pauses while the visitor is in the box or has typed.
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    const timer = setInterval(() => {
      const paused = { focused: document.activeElement === box.current, hasText: !!box.current?.value, reducedMotion: reduce.matches };
      setExample((i) => nextExample(i, paused));
    }, ROTATE_MS);
    return () => clearInterval(timer);
  }, []);

  // Bring Becky's progress into view when a question starts (it is below the fold on phones).
  const working = state.kind === "working";
  useEffect(() => {
    if (!working) return;
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    results.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "nearest" });
  }, [working]);

  // When the answer arrives, move focus to it so screen readers start reading there.
  const finished = state.kind === "done";
  useEffect(() => {
    if (finished) answerHeading.current?.focus();
  }, [finished]);

  const trimmed = question.trim();
  const tooShort = trimmed.length < MIN_CHARS;

  // An example button fills the box (never sends it) and brings the box into view.
  const pick = (text: string, block: ScrollLogicalPosition) => {
    setQuestion(text);
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    box.current?.focus({ preventScroll: true });
    box.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block });
  };

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (tooShort || working) return;
    controller.current?.abort();
    const ctrl = new AbortController();
    controller.current = ctrl;
    setState({ kind: "working" });
    setSteps(initialSteps());
    try {
      // Functional updates: several events can arrive in one chunk and must all apply.
      const onStage = (stage: StageEvent) => {
        if (!ctrl.signal.aborted) setSteps((s) => applyStage(s, stage));
      };
      const result = await askStream(API_BASE, trimmed, onStage, ctrl.signal);
      setSteps(finishSteps);
      setState({ kind: "done", result });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setSteps(failSteps);
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
            ref={box}
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
            placeholder={`e.g. ${EXAMPLES[example].question}`}
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
          <ExampleChips id="try-label" label="Not sure what to ask? Try one of these:" disabled={working} onPick={(t) => pick(t, "nearest")} />
        </form>
      </section>

      <div ref={results} className="results">
        {state.kind === "working" && (
          <section className="result working">
            <Becky pose="working" width={140} />
            <div className="steps-col">
              <h2>Becky is reading the abstracts…</h2>
              <Timeline steps={steps} variant="list" />
              <p className="sr-only" role="status">
                {nowText(steps)}
              </p>
            </div>
          </section>
        )}
        {state.kind === "failed" && (
          <section className={hasProgress(steps) ? "result working" : "result"}>
            {hasProgress(steps) && <Becky pose="noAnswer" width={140} />}
            <div className="steps-col">
              {hasProgress(steps) && <Timeline steps={steps} variant="list" />}
              <div className="error-box" role="alert">
                {state.message}
              </div>
            </div>
          </section>
        )}
        {state.kind === "done" && (
          <Answer result={state.result} steps={steps} headingRef={answerHeading} onPick={(t) => pick(t, "center")} />
        )}
      </div>
    </>
  );
}

// The first BUTTON_COUNT examples as buttons, under the question box and again after a refusal.
function ExampleChips({ id, label, disabled, onPick }: { id: string; label: string; disabled: boolean; onPick: (text: string) => void }) {
  return (
    <>
      <div className="try" id={id}>
        {label}
      </div>
      <div className="chips" role="group" aria-labelledby={id}>
        {EXAMPLES.slice(0, BUTTON_COUNT).map(({ question: text }) => (
          <button key={text} type="button" className="chip" disabled={disabled} onClick={() => onPick(text)}>
            {text}
          </button>
        ))}
      </div>
    </>
  );
}

// Live-region text: the step in progress and its server message, e.g. "Synthesizing: writing a cited answer…".
function nowText(steps: Step[]): string {
  const now = steps.find((s) => s.status === "running");
  return now ? `${now.label}${now.meta ? `: ${now.meta}` : ""}` : "Starting";
}

type AnswerProps = {
  result: QueryResponse;
  steps: Step[];
  headingRef: React.Ref<HTMLHeadingElement>;
  onPick: (text: string) => void;
};

function Answer({ result, steps, headingRef, onPick }: AnswerProps) {
  const views = sourceViews(result);
  const numberOf = new Map(views.map((v) => [v.id, v.n]));
  const [open, setOpen] = useState<Set<number>>(() => new Set());
  const [jump, setJump] = useState<{ n: number; at: number } | null>(null);

  // A clicked [n]: scroll to its card, focus it and play the glow (skipped with reduced motion; the blue border stays).
  useEffect(() => {
    if (!jump) return;
    const card = document.getElementById(anchorId(jump.n));
    if (!card) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    card.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    card.focus({ preventScroll: true });
    if (!reduce && typeof card.animate === "function") {
      card.animate(
        [{ boxShadow: "0 0 0 0 rgba(79,134,232,0)" }, { boxShadow: "0 0 0 6px rgba(79,134,232,.35)", offset: 0.15 }, { boxShadow: "0 0 0 0 rgba(79,134,232,0)" }],
        { duration: 2200, easing: "ease-out" },
      );
    }
  }, [jump]);

  const openSource = (n: number) => {
    setOpen((s) => new Set(s).add(n));
    setJump({ n, at: Date.now() });
  };
  const toggle = (n: number) =>
    setOpen((s) => {
      const next = new Set(s);
      if (!next.delete(n)) next.add(n);
      return next;
    });

  const list = (items: Claim[]) => (
    <ol className="findings">
      {items.map((c) => (
        <li key={c.id}>
          {c.text}
          {[...new Set(c.citation_ids)].map((id) => {
            const n = numberOf.get(id)!;
            return (
              <a
                key={id}
                className="cite-link"
                href={`#${anchorId(n)}`}
                aria-label={`Source ${n}`}
                onClick={(e) => {
                  e.preventDefault();
                  openSource(n);
                }}
              >
                [{n}]
              </a>
            );
          })}
        </li>
      ))}
    </ol>
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
      {views.length > 0 && (
        <Sources
          views={views}
          open={open}
          marked={jump?.n ?? null}
          onToggle={toggle}
          countLine={countText(result.citations.length, views.filter((v) => v.citation).length)}
        />
      )}
    </>
  );

  if (result.abstain_reason === "personal_advice") {
    return (
      <section className="result">
        <div className="no-answer">
          <Becky pose="noAnswer" width={140} />
          <div className="refusal">
            <h2 ref={headingRef} tabIndex={-1} className="refusal-title">
              {ADVICE_TITLE}
            </h2>
            {ADVICE_TEXT}
          </div>
        </div>
        <div className="try-instead">
          <h3>Try this instead</h3>
          <ul>
            <li>{ADVICE_TIP}</li>
          </ul>
          <ExampleChips id="retry-label" label="Or try one she can answer:" disabled={false} onPick={onPick} />
        </div>
      </section>
    );
  }
  if (result.abstained) {
    return (
      <section className="result">
        <Timeline steps={steps} variant="trail" />
        <div className="no-answer">
          <Becky pose="noAnswer" width={140} />
          <div className="refusal">
            <h2 ref={headingRef} tabIndex={-1} className="refusal-title">
              Not enough evidence
            </h2>
            {NO_ANSWER_TEXT[result.abstain_reason ?? ""] ?? NO_ANSWER_FALLBACK}
          </div>
        </div>
        <div className="try-instead">
          <h3>Try this instead</h3>
          <ul>
            <li>Ask a yes / no question about what a study found.</li>
            <li>
              Becky has only read {CORPUS_ABSTRACTS.toLocaleString("en-US")} PubMed abstracts, so many topics aren&apos;t there.
            </li>
          </ul>
          <ExampleChips id="retry-label" label="Or try one she can answer:" disabled={false} onPick={onPick} />
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
      <Timeline steps={steps} variant="trail" />
      <h2 ref={headingRef} tabIndex={-1}>
        Becky&apos;s answer
      </h2>
      <div className="verdict">{VERDICT_LABEL[result.verdict] ?? result.verdict}</div>
      {findings}
    </section>
  );
}
