import { EVAL_FILES, formatRefs, resolve, type EvalData, type Format, type Ref } from "@/lib/evalFormat";

// /eval report from the approved mockup (docs/design/m5-eval-mockup.html). Every number goes through <N>,
// which writes its source refs into data-src so scripts/check-eval-page.ts can verify the built page.

type Sys = "baseline" | "agent" | "closed_book";
const SYSTEMS: { key: Sys; name: string; color: string }[] = [
  { key: "baseline", name: "Baseline", color: "#3A6CC6" },
  { key: "agent", name: "Agent", color: "#C2562E" },
  { key: "closed_book", name: "Closed-book", color: "#7A7F8C" },
];
const T2 = ["baseline", "agent"] as const;
const ANSWER_ROWS = [
  { key: "accuracy", label: "Overall accuracy", chartLabel: "Overall accuracy" },
  { key: "coverage", label: "Coverage", chartLabel: "Coverage (answered)" },
  { key: "answered_only_accuracy", label: "Accuracy when answered", chartLabel: "Accuracy when answered" },
] as const;

export default function EvalReport({ data }: { data: EvalData }) {
  // `svg`: inside an SVG <text>, where only <tspan> is valid.
  const N: NFn = ({ fmt, refs, svg }) => {
    const Tag = svg ? "tspan" : "span";
    return (
      <Tag data-src={refs.join("|")} data-fmt={fmt}>
        {formatRefs(data, fmt, refs)}
      </Tag>
    );
  };
  const num = (ref: Ref) => resolve(data, ref) as number;
  const sys = (s: Sys, rest: string) => `summary:systems.${s}.${rest}` as Ref;
  const frac = (ref: string) => <N fmt="frac" refs={[`${ref}.count` as Ref, `${ref}.n` as Ref]} />;
  const n = <N fmt="int" refs={["summary:n_questions"]} />;
  const cl = "claims:agent_minus_baseline";
  const t2 = (s: "baseline" | "agent", rest: string) => `t2_score:systems.${s}.${rest}` as Ref;
  const pv = "t2_score:agent_vs_baseline";
  const rt = "t2_score:agent_retry";

  return (
    <>
      <h1>How GroundedMed is built and tested</h1>
      <p className="lede">
        An agentic RAG pipeline that answers research questions from medical abstracts, with every finding cited and checked, tested on two
        hash-locked PubMedQA test sets with the decision rules written before the runs. Every result below is read from the stored result files.
      </p>

      <section className="eval-card" aria-labelledby="h-built">
        <h2 id="h-built">What it does</h2>
        <ul>
          <li>
            <b>Planner</b> (AI) turns the question into a few search queries.
          </li>
          <li>
            <b>Retriever</b> (code) searches <N fmt="int" refs={["corpus:count"]} /> abstracts by meaning, reranks the best matches and keeps the
            top 5. If even the best match is weak, an <b>evidence gate</b> says &quot;not enough evidence&quot; instead of answering.
          </li>
          <li>
            <b>Synthesizer</b> (AI) writes a yes / no / maybe verdict as short findings, each citing one of those abstracts.
          </li>
          <li>
            <b>Critic</b> (AI) checks each finding against the abstract it cites. It can remove findings or hold back the verdict, but never add
            anything. When it flags an answer that would otherwise be refused, the synthesizer gets one guided rewrite with the critic&apos;s notes,
            and the critic checks that once.
          </li>
          <li>
            <b>Validator</b> (code) drops any finding whose citation is not one of the abstracts retrieved for that question, so every finding on
            screen cites a source you can open.
          </li>
        </ul>
      </section>

      <section className="eval-card" aria-labelledby="h-method">
        <h2 id="h-method">How it&apos;s tested</h2>
        <ul>
          <li>
            <b>Locked test sets, each run once.</b> <N fmt="int" refs={["summary:n_questions"]} /> test questions were frozen (IDs and hash)
            before any system was built. Later, a fresh set of <N fmt="int" refs={["t2_score:n_questions"]} /> was locked the same way from
            questions nobody had looked at, before any tuning. Baseline and agent ran once on each set (the closed-book control on the first);
            everything tuned was tuned on separate dev questions.
          </li>
          <li>
            <b>Rules before results.</b> The decision rules for the latest test were written down and coded into its scorer before it ran,
            and code blocks a second run on a locked set unless overridden with a recorded reason (never used).
          </li>
          <li>
            <b>Refusals count as wrong</b>, so refusing hard questions can&apos;t look like skill.
          </li>
          <li>
            <b>Paired statistics.</b> Wilson 95% ranges for each system&apos;s percentages; for differences, a paired bootstrap over the same
            questions (<N fmt="plain" refs={["t2_score:agent_vs_baseline.bootstrap.resamples"]} /> resamples) and an exact sign test. Claim
            support uses a bootstrap over questions; time and cost a bootstrap.
          </li>
          <li>
            <b>Independent judge (v1.0.0).</b> An AI from a different company (<N fmt="text" refs={["judge_config:judge_model"]} />, Google),
            never the system&apos;s own critic, rated each finding against the abstract it cites, blinded to which system wrote it.
          </li>
          <li>
            All figures are produced by the evaluation code and shown here unchanged, only rounded for display (percentages to one decimal,
            seconds to two, dollars to four; a value very close to zero keeps a second decimal so its sign stays visible).
          </li>
        </ul>
      </section>

      <section className="eval-card" aria-labelledby="h-latest">
        <h2 id="h-latest">
          Latest test (v1.1.0): <N fmt="int" refs={["t2_score:n_questions"]} /> fresh questions
        </h2>
        <p className="soft">
          Run on <N fmt="date" refs={["t2_baseline_config:created_utc", "t2_agent_config:created_utc"]} />, scored on{" "}
          <N fmt="date" refs={["t2_config:created_utc"]} />. Accuracy is against PubMedQA&apos;s own yes / no / maybe answers; a refusal counts
          as wrong.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <td />
                <th scope="col">Baseline</th>
                <th scope="col">Agent</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Correct answers</th>
                {T2.map((s) => (
                  <td key={s}>
                    <b>
                      <N fmt="pctP" refs={[t2(s, "accuracy_pct")]} />
                    </b>{" "}
                    <span className="n">
                      <N fmt="frac" refs={[t2(s, "correct"), t2(s, "n_questions")]} />
                    </span>
                    <br />
                    <span className="ci">
                      95%: <N fmt="pctPRange" refs={[t2(s, "accuracy_wilson95_pct")]} />
                    </span>
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Refused (&quot;not enough evidence&quot;)</th>
                {T2.map((s) => (
                  <td key={s}>
                    <N fmt="int" refs={[t2(s, "refused")]} />
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Hit@1 / Hit@5 (source paper ranked first / in the top 5)</th>
                {T2.map((s) => (
                  <td key={s}>
                    <N fmt="pct" refs={[`t2_${s}:hit_at_1` as Ref]} /> / <N fmt="pct" refs={[`t2_${s}:hit_at_5` as Ref]} />
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Typical time (median)</th>
                {T2.map((s) => (
                  <td key={s}>
                    <N fmt="secs" refs={[t2(s, "median_time_per_answer_ms")]} />
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">AI cost per question (mean)</th>
                {T2.map((s) => (
                  <td key={s}>
                    <N fmt="usd" refs={[t2(s, "cost_usd_per_question")]} />
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          The guided rewrite ran on <N fmt="int" refs={[`${rt}.questions_retried`]} /> of the <N fmt="int" refs={["t2_score:n_questions"]} />{" "}
          questions, all of which would otherwise have been refused; <N fmt="int" refs={[`${rt}.after_retry_right`]} /> of them ended with the
          right answer.
        </p>
        <p className="soft small">
          Agent and baseline were statistically tied: paired difference <N fmt="ptsP" refs={[`${pv}.accuracy_diff_agent_minus_baseline_points`]} />{" "}
          (95% range <N fmt="ptsPRange" refs={[`${pv}.accuracy_diff_bootstrap95_points`]} />
          ), exact sign test p = <N fmt="p2" refs={[`${pv}.sign_test_p_exact_two_sided`]} />. Measured on a developer laptop, one question at a
          time; cost = Claude tokens at list price only. This test had no closed-book or AI-judge run.
        </p>
      </section>

      <h2 className="detail-title">
        First test (v1.0.0): <N fmt="int" refs={["summary:n_questions"]} /> questions
      </h2>
      <p className="soft">
        The agent as it was at v1.0.0, with a closed-book control and the independent AI judge. Systems tested on{" "}
        <N fmt="date" refs={["baseline_config:created_utc", "agent_config:created_utc", "closed_config:created_utc"]} />, scored on{" "}
        <N fmt="date" refs={["final_config:created_utc"]} /> (scoring code <N fmt="commit" refs={["final_config:git_commit"]} />
        ).
      </p>

      <section className="eval-card" aria-labelledby="h-retrieval">
        <h2 id="h-retrieval">Reading the abstracts is what makes the answers good</h2>
        <p className="soft">
          Share of the {n} test questions answered correctly (PubMedQA&apos;s own yes / no / maybe answers; a refusal counts as wrong).
        </p>
        <div className="tiles">
          {SYSTEMS.map(({ key, name }) => (
            <div key={key} className={key === "closed_book" ? "tile closed" : "tile grounded"}>
              <span>{key === "closed_book" ? "Closed-book (same AI, no abstracts)" : `${name} (reads abstracts)`}</span>
              <b>
                <N fmt="pct" refs={[sys(key, "accuracy.value")]} />
              </b>
              <em>{frac(sys(key, "accuracy"))}</em>
            </div>
          ))}
        </div>
        <div className="gains">
          {(["baseline", "agent"] as const).map((s) => (
            <p key={s}>
              <b className="gain">
                <N fmt="pts" refs={[`summary:comparisons.${s}_minus_closed_book.accuracy.estimate`]} />
              </b>{" "}
              {s} vs closed-book{" "}
              <span className="ci">
                (95% range <N fmt="ptsRange" refs={[`summary:comparisons.${s}_minus_closed_book.accuracy.ci95`]} />)
              </span>
            </p>
          ))}
        </div>
        <p className="soft small">
          Without the abstracts, the AI did no better than always answering &quot;yes&quot; (
          <N fmt="pct" refs={["summary:majority_class_reference.value"]} />
          ). Both gains are well clear of zero.
        </p>
      </section>

      <section className="eval-card" aria-labelledby="h-trade">
        <h2 id="h-trade">What the agent&apos;s critic adds</h2>
        <p>
          The agent adds a planner and a critic on top of the baseline. The critic checks every finding and limitation against the abstract it
          cites, removes what it can&apos;t back up, and holds back a verdict its findings don&apos;t support. On the same {n} questions, an
          independent AI judge from a different company rated the agent&apos;s limitations fully supported more often.
        </p>
        <div className="trade">
          <div className="side plus">
            <h3>Better-supported answers</h3>
            <ul>
              <li>
                <b>Limitations fully supported:</b> <N fmt="pct" refs={["claims:systems.agent.limitation_strict.value"]} /> vs{" "}
                <N fmt="pct" refs={["claims:systems.baseline.limitation_strict.value"]} />: <N fmt="pts" refs={[`${cl}.limitation_strict.estimate`]} />{" "}
                <span className="ci">
                  (95% range <N fmt="ptsRange" refs={[`${cl}.limitation_strict.ci95`]} />; clear)
                </span>
              </li>
              <li>
                <b>Findings fully supported:</b> <N fmt="pct" refs={["claims:systems.agent.claim_strict.value"]} /> vs{" "}
                <N fmt="pct" refs={["claims:systems.baseline.claim_strict.value"]} />: <N fmt="pts" refs={[`${cl}.claim_strict.estimate`]} />{" "}
                <span className="ci">
                  (95% range <N fmt="ptsRange" refs={[`${cl}.claim_strict.ci95`]} />; just includes 0, so no clear difference)
                </span>
              </li>
              <li>
                <b>Answers with any item not fully supported:</b> {frac("claims:systems.agent.questions_with_a_not_supported_item")} (
                <N fmt="pct" refs={["claims:systems.agent.questions_with_a_not_supported_item.value"]} />) vs{" "}
                {frac("claims:systems.baseline.questions_with_a_not_supported_item")} (
                <N fmt="pct" refs={["claims:systems.baseline.questions_with_a_not_supported_item.value"]} />)
              </li>
            </ul>
          </div>
          <div className="side neutral">
            <h3>Agent and baseline side by side</h3>
            <ul>
              <li>
                <b>Overall accuracy:</b> <N fmt="pct" refs={[sys("agent", "accuracy.value")]} /> vs{" "}
                <N fmt="pct" refs={[sys("baseline", "accuracy.value")]} />
              </li>
              <li>
                <b>Questions answered:</b> <N fmt="pct" refs={[sys("agent", "coverage.value")]} /> vs{" "}
                <N fmt="pct" refs={[sys("baseline", "coverage.value")]} />; the critic holds back a verdict when the cited evidence doesn&apos;t
                support it
              </li>
              <li>
                <b>Accuracy on the questions each answered:</b> <N fmt="pct" refs={[sys("agent", "answered_only_accuracy.value")]} /> vs{" "}
                <N fmt="pct" refs={[sys("baseline", "answered_only_accuracy.value")]} />
              </li>
              <li>
                <b>Time:</b> typical answer <N fmt="secs" refs={[sys("agent", "latency_ms.p50.estimate")]} /> vs{" "}
                <N fmt="secs" refs={[sys("baseline", "latency_ms.p50.estimate")]} />
              </li>
              <li>
                <b>AI cost per question:</b> <N fmt="usd" refs={[sys("agent", "llm_cost_only_usd_per_query.mean.estimate")]} /> vs{" "}
                <N fmt="usd" refs={[sys("baseline", "llm_cost_only_usd_per_query.mean.estimate")]} />
              </li>
            </ul>
          </div>
        </div>
        <p className="soft small">
          Support figures are the AI judge&apos;s estimate; the second-AI cross-check below tests the judge itself.
        </p>
      </section>

      <section className="eval-card" aria-labelledby="h-accuracy">
        <h2 id="h-accuracy">Answer accuracy against PubMedQA&apos;s answers</h2>
        <p className="soft">
          This is ground-truth accuracy: does the yes / no / maybe verdict match PubMedQA&apos;s label? A refusal counts as wrong. Coverage is
          always shown next to accuracy-when-answered, because refusing hard questions would otherwise look like skill.
        </p>
        <div className="legend" aria-hidden="true">
          {SYSTEMS.map(({ key, name, color }) => (
            <span key={key}>
              <i style={{ background: color }} />
              {name}
            </span>
          ))}
          <span>
            <i className="refkey" />
            always answer &quot;yes&quot; (reference)
          </span>
        </div>
        <DotChart data={data} num={num} N={N} />
        <DotChart data={data} num={num} N={N} narrow />
        <p className="soft small">Dot = result; line = 95% range (Wilson). Axis starts at 40% so the ranges are readable. Exact values are in the table.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <td />
                {SYSTEMS.map(({ key, name }) => (
                  <th key={key} scope="col">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ANSWER_ROWS.map(({ key, label }) => (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  {SYSTEMS.map(({ key: s }) => (
                    <td key={s}>
                      <b>
                        <N fmt="pct" refs={[sys(s, `${key}.value`)]} />
                      </b>{" "}
                      <span className="n">{frac(sys(s, key))}</span>
                      <br />
                      <span className="ci">
                        95%: <N fmt="pctRange" refs={[sys(s, `${key}.wilson95`)]} />
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="soft small">
          Closed-book = the same AI with no abstracts at all; it scored about the same as always saying &quot;yes&quot; (
          <N fmt="pct" refs={["summary:majority_class_reference.value"]} />
          ), so the abstracts matter.
        </p>
      </section>

      <section className="eval-card" aria-labelledby="h-retrieval-detail">
        <h2 id="h-retrieval-detail">Finding the right abstract</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <td />
                <th scope="col">Baseline</th>
                <th scope="col">Agent</th>
              </tr>
            </thead>
            <tbody>
              {(
                [
                  ["hit_at_1", "Hit@1 (source paper ranked first)"],
                  ["hit_at_5", "Hit@5 (source paper in the top 5)"],
                ] as const
              ).map(([key, label]) => (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  {(["baseline", "agent"] as const).map((s) => (
                    <td key={s}>
                      <N fmt="pct" refs={[sys(s, `${key}.value`)]} /> <span className="n">{frac(sys(s, key))}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="soft small">These are favourable by design of the benchmark; see Limitations.</p>
      </section>

      <section className="eval-card" aria-labelledby="h-speed">
        <h2 id="h-speed">Speed and cost</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <td />
                {SYSTEMS.map(({ key, name }) => (
                  <th key={key} scope="col">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(
                [
                  ["Typical time (p50)", "latency_ms.p50", "secs", "secsRange"],
                  ["Slow case (p95)", "latency_ms.p95", "secs", "secsRange"],
                  ["AI cost per question (mean)", "llm_cost_only_usd_per_query.mean", "usd", "usdRange"],
                ] as const
              ).map(([label, path, fmt, rangeFmt]) => (
                <tr key={path}>
                  <th scope="row">{label}</th>
                  {SYSTEMS.map(({ key: s }) => (
                    <td key={s}>
                      <N fmt={fmt} refs={[sys(s, `${path}.estimate`)]} />
                      <br />
                      <span className="ci">
                        95%: <N fmt={rangeFmt} refs={[sys(s, `${path}.ci95`)]} />
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="soft small">
          Measured on a developer laptop, one question at a time (not production load). Cost = Claude tokens at list price only; search and
          reranking were on a free allowance and are not included. Ranges: bootstrap 95%.
        </p>
      </section>

      <section className="eval-card" aria-labelledby="h-support">
        <h2 id="h-support">
          Are the findings backed by their sources? <span className="badge">AI-judge estimate</span>
        </h2>
        <p className="soft">
          Not the same thing as the accuracy above. An independent AI judge (<N fmt="text" refs={["judge_config:judge_model"]} />, Google; not the
          AI that writes the answers) read each finding with the abstract it cites and labelled it fully, partly or not supported. We report{" "}
          <b>fully supported</b> only.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <td />
                <th scope="col">Baseline</th>
                <th scope="col">Agent</th>
                <th scope="col">Agent minus baseline</th>
              </tr>
            </thead>
            <tbody>
              {(
                [
                  ["claim_strict", "Findings fully supported"],
                  ["limitation_strict", "Limitations fully supported"],
                ] as const
              ).map(([key, label]) => (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  {(["baseline", "agent"] as const).map((s) => (
                    <td key={s}>
                      <b>
                        <N fmt="pct" refs={[`claims:systems.${s}.${key}.value`]} />
                      </b>{" "}
                      <span className="n">{frac(`claims:systems.${s}.${key}`)}</span>
                      <br />
                      <span className="ci">
                        95%: <N fmt="pctRange" refs={[`claims:systems.${s}.${key}.question_bootstrap95`]} />
                      </span>
                    </td>
                  ))}
                  <td>
                    <N fmt="pts" refs={[`${cl}.${key}.estimate`]} />
                    <br />
                    <span className="ci">
                      <N fmt="ptsRange" refs={[`${cl}.${key}.ci95`]} />
                    </span>
                  </td>
                </tr>
              ))}
              <tr>
                <th scope="row">Answers with at least one item not fully supported</th>
                {(["baseline", "agent"] as const).map((s) => (
                  <td key={s}>{frac(`claims:systems.${s}.questions_with_a_not_supported_item`)}</td>
                ))}
                <td>—</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="soft small">
          The agent&apos;s critic removes items it can&apos;t back up and holds back unsupported verdicts, so the two systems are judged on
          different numbers of items. The findings difference is within noise: its range just includes 0.
        </p>
        <h3>Second-AI cross-check</h3>
        <p>
          To test the judge, a second AI from a different company (ChatGPT, OpenAI) labelled a blind sample of <Sample N={N} /> of the same
          items without seeing the judge&apos;s labels. It agreed with the judge on <Agree group="judge_supported" N={N} /> of the items the
          judge accepted, and on <Agree group="judge_rejected" N={N} /> of the items the judge rejected. It rejected some items the judge had
          accepted, which suggests the judge&apos;s figures may be <b>optimistic</b>. The sample held findings only, no limitations. Both are
          AIs and can share blind spots, so agreement does not prove the judge right; how ChatGPT was run is as reported by the project
          owner. <b>No person checked these items.</b>
        </p>
      </section>

      <section className="eval-card" aria-labelledby="h-limits">
        <h2 id="h-limits">Limitations</h2>
        <ul>
          <li>
            PubMedQA questions are often written from the article&apos;s title, so finding the right abstract is easier than real-world search.
            We left titles out of the searchable text and added <N fmt="int" refs={["corpus:source_counts.pqa_u"]} /> similar-topic abstracts as
            distractors, but Hit@5 is still <N fmt="pct" refs={[sys("baseline", "hit_at_5.value")]} /> on the first test and{" "}
            <N fmt="pct" refs={["t2_baseline:hit_at_5"]} /> on the latest.
          </li>
          <li>
            PubMedQA has no sentence-level labels showing which sentence supports an answer, so &quot;supported&quot; is judged by AI, not
            checked against human annotations.
          </li>
          <li>
            The claim-support judge is an AI (a preview model) and was cross-checked only by another AI on <Sample N={N} /> findings; its
            limitation labels were not cross-checked. The judge ran on the first test (v1.0.0) only.
          </li>
          <li>
            One run per system on each test (<N fmt="int" refs={["summary:n_questions"]} /> and <N fmt="int" refs={["t2_score:n_questions"]} />{" "}
            questions): small differences are within the ranges shown.
          </li>
          <li>
            PubMedQA is public, so the AI may have seen it in training; the closed-book result suggests it does not simply remember the
            answers.
          </li>
          <li>Times are from a developer laptop; costs exclude search and reranking.</li>
          <li>
            The live demo also screens out questions asking for personal medical advice before Becky answers. That screen is not part of
            either test.
          </li>
        </ul>
      </section>

      <footer className="data">
        Data: latest test <code>{EVAL_FILES.t2_score}</code>, <code>{EVAL_FILES.t2_baseline}</code> and <code>{EVAL_FILES.t2_agent}</code>;
        first test <code>{EVAL_FILES.summary}</code> (commit <N fmt="commit" refs={["final_config:git_commit"]} />), <code>{EVAL_FILES.claims}</code>{" "}
        and <code>agreement.json</code> (commit <N fmt="commit" refs={["cross_config:git_commit"]} />
        ). Read when the page is built; never recalculated.
      </footer>
    </>
  );
}

type NFn = (props: { fmt: Format; refs: Ref[]; svg?: boolean }) => React.ReactElement;

// Size of the blind sample: judge-accepted rows + judge-rejected rows.
function Sample({ N }: { N: NFn }) {
  return <N fmt="sum" refs={["agreement:judge_supported.pooled.rows", "agreement:judge_rejected.pooled.rows"]} />;
}

function Agree({ group, N }: { group: "judge_supported" | "judge_rejected"; N: NFn }) {
  const p = `agreement:${group}.pooled`;
  return (
    <>
      <b>
        <N fmt="frac" refs={[`${p}.agree` as Ref, `${p}.compared` as Ref]} />
      </b>{" "}
      = <N fmt="pct" refs={[`${p}.agreement` as Ref]} /> (95%: <N fmt="pctRange" refs={[`${p}.wilson95` as Ref]} />)
    </>
  );
}

// Dots = estimate, line = Wilson 95% range; x axis 40-100% (dots, not bars, so the axis need not start at 0).
// `narrow`: phone layout (CSS shows one of the two), labels above each row so the text stays readable.
function DotChart({ data, num, N, narrow }: { data: EvalData; num: (r: Ref) => number; N: NFn; narrow?: boolean }) {
  const [w, left, right, rowH, top] = narrow ? [340, 12, 16, 92, 30] : [640, 190, 20, 74, 30];
  const labelDy = narrow ? -14 : rowH / 2 - 4;
  const dotsTop = narrow ? 8 : 12;
  const x = (v: number) => left + ((v - 0.4) / 0.6) * (w - left - right);
  const h = top + rowH * ANSWER_ROWS.length + 30;
  const ref = num("summary:majority_class_reference.value");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label="Dot chart of overall accuracy, coverage and accuracy when answered for the three systems, with their ranges; exact values are in the table below"
      className={narrow ? "chart chart-narrow" : "chart chart-wide"}
    >
      {[40, 50, 60, 70, 80, 90, 100].map((t) => (
        <g key={t}>
          <line x1={x(t / 100)} y1={top - 10} x2={x(t / 100)} y2={h - 28} className="grid" />
          <text x={x(t / 100)} y={h - 10} className="tick">
            {`${t}%`}
          </text>
        </g>
      ))}
      {ANSWER_ROWS.map(({ key, chartLabel }, i) => {
        const y0 = top + i * rowH;
        return (
          <g key={key}>
            <text x={0} y={y0 + labelDy + (narrow ? 22 : 0)} className="rowlabel">
              {chartLabel}
            </text>
            {key === "accuracy" && (
              <>
                <line x1={x(ref)} y1={y0 + (narrow ? 22 : 4)} x2={x(ref)} y2={y0 + rowH - 14} className="ref" />
                <text x={x(ref) + 4} y={y0 + rowH - 14} className="reflabel">
                  always &quot;yes&quot; <N svg fmt="pct" refs={["summary:majority_class_reference.value"]} />
                </text>
              </>
            )}
            {SYSTEMS.map(({ key: s, color }, j) => {
              const m = resolve(data, `summary:systems.${s}.${key}`) as { value: number; wilson95: [number, number] };
              const yy = y0 + dotsTop + (narrow ? 22 : 0) + j * 16;
              return (
                <g key={s}>
                  <line x1={x(m.wilson95[0])} y1={yy} x2={x(m.wilson95[1])} y2={yy} stroke={color} strokeWidth={2.5} />
                  <circle cx={x(m.value)} cy={yy} r={5.5} fill={color} />
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
