import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { checkEvalHtml } from "./checkEval.ts";
import { loadEval } from "./evalData.ts";
import { formatRefs, resolve, type Format, type Ref } from "./evalFormat.ts";

// A small page in the built format (React's <span data-src data-fmt>), made from the real stored values.
const data = loadEval()!;
const n = (fmt: Format, ...refs: Ref[]) => `<span data-src="${refs.join("|")}" data-fmt="${fmt}">${formatRefs(data, fmt, refs)}</span>`;
const WORDING = "Second-AI cross-check No person checked these items. fully supported AI-judge estimate";
const page = (body: string) =>
  `<html><main><p>${n("pct", "summary:systems.baseline.accuracy.value")} ${n("frac", "summary:systems.agent.accuracy.count", "summary:systems.agent.accuracy.n")}` +
  ` (95% range ${n("ptsRange", "summary:comparisons.agent_minus_baseline.accuracy.ci95")}) Hit@5 (source paper in the top 5)</p>` +
  `<svg aria-label="Dot chart with their ranges"><text class="tick">40%</text></svg><p>${WORDING}</p>${body}</main></html>`;
const check = (html: string) => checkEvalHtml(html, data, 3).problems;

test("a correct page passes", () => {
  assert.deepEqual(check(page("")), []);
});

// Deliberately broken pages: each must be caught.
const SABOTAGES: [string, (h: string) => string, RegExp][] = [
  ["wrong text in a sourced number", (h) => h.replace(">76.5%<", ">77.5%<"), /page shows "77.5%"/],
  ["hard-coded range next to '95% range'", (h) => h.replace(page(""), page("<p>95% range +3.8 to +9.6</p>")), /unsourced number/],
  ["hard-coded range after '95%:'", (h) => h.replace(page(""), page("<p>95%: 70.2–81.8%</p>")), /unsourced number/],
  ["hard-coded integers, points and seconds", (h) => h.replace(page(""), page("<p>12/20 agreed, +3 pts, 12 s</p>")), /unsourced number/],
  ["number in the chart's aria-label", (h) => h.replace("Dot chart with their ranges", "Dot chart, agent 99.9%"), /number in aria-label/],
  ["lenient claim support behind a correct-looking number", (h) => h.replace(page(""), page(n("pct", "claims:systems.agent.claim_lenient.value"))), /lenient claim support/],
  ["claim-support Wilson range instead of the question bootstrap", (h) => h.replace(page(""), page(n("pctRange", "claims:systems.agent.claim_strict.wilson95"))), /question bootstrap/],
  ["a number dressed as an axis tick", (h) => h.replace('<text class="tick">40%</text>', '<text class="tick">99.9%</text>'), /unexpected axis tick/],
  ["'at least partly supported' wording", (h) => h.replace(WORDING, `${WORDING} at least partly supported`), /forbidden wording/],
  ["claiming a person checked the items", (h) => h.replace("No person checked these items.", "These items were manually checked."), /forbidden wording .*manual/],
  ["a ref that does not exist", (h) => h.replace("accuracy.ci95", "accuracy.ci99"), /missing summary:/],
];
for (const [name, sabotage, expected] of SABOTAGES) {
  test(`checker catches: ${name}`, () => {
    const problems = check(sabotage(page("")));
    assert.ok(problems.some((p) => expected.test(p)), `expected ${expected} in ${JSON.stringify(problems)}`);
  });
}

test("the page's fixed reading words are checked against the data", () => {
  const flipped = structuredClone(data) as typeof data & { claims: { agent_minus_baseline: { claim_strict: { includes_zero: boolean } } } };
  flipped.claims.agent_minus_baseline.claim_strict.includes_zero = false;
  assert.ok(checkEvalHtml(page(""), flipped, 3).problems.some((p) => /findings: no clear difference/.test(p)));
});

test("the cross-check sample held findings only (the page says its limitation labels were not cross-checked)", () => {
  const csv = readFileSync(new URL("../../eval/audits/judge-test-20261026-091111-06b80ea-second-ai-crosscheck.csv", import.meta.url), "utf8");
  const [header, ...rows] = parseCsv(csv.replace(/^\uFEFF/, ""));
  const ids = rows.filter((r) => r.length > 1).map((r) => r[header.indexOf("claim_id")]);
  assert.equal(ids.length, 30);
  assert.ok(ids.every((id) => /^c\d+$/.test(id)), JSON.stringify(ids));
});

test("commit IDs on the page are checked through the commit map", () => {
  const recorded = resolve(data, "final_config:git_commit") as string;
  const current = "c".repeat(40);
  const mapped = { ...data, commit_map: { commits: { [recorded]: current } } };
  const html = page(`<span data-src="final_config:git_commit" data-fmt="commit">ccccccc, recorded as ${recorded.slice(0, 7)}</span>`);
  assert.deepEqual(checkEvalHtml(html, mapped, 3).problems, []);
  assert.ok(checkEvalHtml(html.replace("ccccccc", "ddddddd"), mapped, 3).problems.some((p) => /page shows/.test(p)));
  assert.ok(checkEvalHtml(html, { ...data, commit_map: { commits: {} } }, 3).problems.some((p) => /no entry/.test(p)));
  const shortened = { ...data, commit_map: { commits: { [recorded]: current, abc1234: "def5678" } } };
  assert.ok(checkEvalHtml(html, shortened, 3).problems.some((p) => /two full IDs/.test(p)));
});

/** Minimal RFC 4180 reader (quoted fields may hold commas, quotes and newlines). */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [[]];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') (field += '"'), i++;
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") rows[rows.length - 1].push(field), (field = "");
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      rows[rows.length - 1].push(field), (field = ""), rows.push([]);
    } else field += ch;
  }
  rows[rows.length - 1].push(field);
  return rows;
}
