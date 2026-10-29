// M5-T5 acceptance check on the built /eval HTML (run by scripts/check-eval-page.ts after `npm run build`).
// 1. every <N> number equals its stored value; 2. no number appears anywhere else (text or attributes);
// 3. refs point at the agreed fields (strict claim support, test split); 4. the page's wording claims
// ("clear", "no clear difference", "less often") still hold for the stored data; 5. ADR-037 wording.

import { FORMATS, formatRefs, parseSrc, resolve, type EvalData, type Format, type Ref } from "./evalFormat.ts";

export const decode = (s: string) =>
  s.replace(/<!-- -->/g, "").replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");

// <span|tspan data-src=".." data-fmt="..">TEXT</span|tspan>  (N never nests, so TEXT has no tags)
const NUM = /<(span|tspan) data-src="([^"]+)" data-fmt="([^"]+)">([^<]*)<\/\1>/g;
export const AXIS_TICKS = ["40%", "50%", "60%", "70%", "80%", "90%", "100%"];
// Fixed words that contain digits but are not results.
const ALLOWED = [/top 5\b/g, /(in|ex)cludes 0\b/g, /95% range/g, /95%:/g, /Wilson 95%/g, /bootstrap 95%/g, /Hit@[15]/g, /p50/g, /p95/g, /Axis starts at 40%/g, /\b[12]\. /g];

export function checkEvalHtml(html: string, data: EvalData, minCount = 100): { count: number; problems: string[] } {
  const problems: string[] = [];
  const start = html.indexOf("<main");
  const main = start < 0 ? "" : html.slice(start, html.indexOf("</main>", start));
  if (!main) return { count: 0, problems: ["no <main> in the page"] };

  let count = 0;
  for (const [, , rawSrc, fmt, text] of main.matchAll(NUM)) {
    count++;
    const refs = parseSrc(decode(rawSrc));
    if (!(fmt in FORMATS)) {
      problems.push(`unknown format ${fmt} for ${rawSrc}`);
      continue;
    }
    for (const r of refs) {
      if (/_lenient/.test(r)) problems.push(`${r}: lenient claim support is never shown (CLAUDE.md)`);
      if (r.startsWith("claims:") && r.includes("wilson95")) problems.push(`${r}: claim-support ranges are the question bootstrap`);
    }
    let expected: string;
    try {
      expected = formatRefs(data, fmt as Format, refs);
    } catch (e) {
      problems.push(String(e));
      continue;
    }
    if (decode(text) !== expected) problems.push(`${rawSrc}: page shows "${decode(text)}", stored value formats as "${expected}"`);
  }
  if (count < minCount) problems.push(`only ${count} sourced numbers found (expected the full report)`);

  // Axis ticks: only the fixed list, nothing else in tick text.
  for (const [, t] of main.matchAll(/<text[^>]*class="tick"[^>]*>([^<]*)<\/text>/g)) {
    if (!AXIS_TICKS.includes(decode(t))) problems.push(`unexpected axis tick "${decode(t)}"`);
  }
  // Attributes: no digits in labels/titles (data-src, coordinates, styles and ids are not shown to people).
  for (const [, name, value] of main.matchAll(/\s(aria-label|title|alt)="([^"]*)"/g)) {
    if (/\d/.test(value)) problems.push(`number in ${name}="${value}"`);
  }
  // Text: after removing sourced numbers, ticks, code and allowed words, no digit may remain.
  let rest = decode(
    main
      .replace(NUM, " ")
      .replace(/<text[^>]*class="tick"[^>]*>[^<]*<\/text>/g, " ")
      .replace(/<code>[^<]*<\/code>/g, " ")
      .replace(/<[^>]+>/g, " "),
  );
  for (const a of ALLOWED) rest = rest.replace(a, " ");
  for (const m of rest.matchAll(/\d/g)) {
    problems.push(`unsourced number near "${rest.slice(Math.max(0, m.index - 30), m.index + 20).replace(/\s+/g, " ")}"`);
    break; // one is enough to fail; the context shows where
  }

  // The page's fixed reading words must still be true of the stored data.
  const truth: [string, boolean][] = [
    ["test split", resolve(data, "summary:split") === "test"],
    ["'limitations: clear' (range excludes 0)", resolve(data, "claims:agent_minus_baseline.limitation_strict.includes_zero") === false],
    ["'findings: no clear difference' (range includes 0)", resolve(data, "claims:agent_minus_baseline.claim_strict.includes_zero") === true],
    ["'accuracy on the questions each answered: no clear difference'", resolve(data, "summary:comparisons.agent_minus_baseline.answered_only_accuracy.includes_zero") === true],
    ["'both gains are well clear of zero'", ["baseline", "agent"].every((s) => (resolve(data, `summary:comparisons.${s}_minus_closed_book.accuracy.ci95` as Ref) as number[])[0] > 0)],
    ["'agreed less often on the agent's accepted findings'", (resolve(data, "agreement:judge_supported.agent.agree") as number) < (resolve(data, "agreement:judge_supported.baseline.agree") as number)],
  ];
  for (const [claim, ok] of truth) if (!ok) problems.push(`page wording no longer true: ${claim}`);

  // Wording rules (CLAUDE.md, ADR-037): no person checked the claims; strict support is the headline.
  const text = decode(main.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ");
  for (const banned of [/manual/i, /human audit/i, /checked by a person/i, /\bhuman\b(?! annotations)/i, /at least partly/i, /lenient/i, /hallucination rate/i]) {
    if (banned.test(text)) problems.push(`forbidden wording ${banned}`);
  }
  for (const needed of ["Second-AI cross-check", "No person checked these items.", "fully supported", "AI-judge estimate"]) {
    if (!text.includes(needed)) problems.push(`missing wording "${needed}"`);
  }
  return { count, problems };
}
