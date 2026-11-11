// /eval page numbers (approved mockup docs/design/m5-eval-mockup.html). Pure, unit-tested.
// Every displayed number is a Ref list + a format name; the page writes both into data-src / data-fmt,
// and scripts/check-eval-page.ts re-reads the stored files to prove each built number matches.

// Stored result files the page reads at build time (repo-relative). Never recalculated on page load.
const FINAL = "eval/results/m4-final-20261026-222211-498b76a";
const CROSS = "eval/results/m4-crosscheck-20261026-233927-865624f";
// v1.1.0: the second frozen test set (test2), scored by eval/runners/score_test2.py. Values there are already percent/points.
const T2_SCORE = "eval/results/m15-test2-score-20261110-225734-bc0d65f";
const T2_BASELINE = "eval/results/test2-baseline-20261110-205943-9553791";
const T2_AGENT = "eval/results/test2-agent-20261110-213115-bd5773d";
export const EVAL_FILES = {
  t2_score: `${T2_SCORE}/summary.json`,
  t2_config: `${T2_SCORE}/run_config.json`,
  t2_baseline: `${T2_BASELINE}/summary.json`,
  t2_agent: `${T2_AGENT}/summary.json`,
  t2_baseline_config: `${T2_BASELINE}/run_config.json`,
  t2_agent_config: `${T2_AGENT}/run_config.json`,
  summary: `${FINAL}/summary.json`,
  final_config: `${FINAL}/run_config.json`,
  claims: `${CROSS}/claim_support.json`,
  agreement: `${CROSS}/agreement.json`,
  cross_config: `${CROSS}/run_config.json`,
  judge_config: "eval/results/judge-test-20261026-091111-06b80ea/run_config.json",
  baseline_config: "eval/results/test-baseline-20261025-194639-4ebd38d/run_config.json",
  agent_config: "eval/results/test-agent-20261025-200618-1f1e727/run_config.json",
  closed_config: "eval/results/test-closed_book-20261025-211502-6eecb38/run_config.json",
  corpus: "data/manifests/corpus.json",
} as const;

// Result files keep the commit IDs they recorded. When a later history rewrite changes the IDs, this optional file
// maps each recorded full ID to the current one, and the page shows both. No file: the recorded ID alone.
export const COMMIT_MAP_FILE = "eval/COMMIT_ID_MAP.json";
export type CommitMap = { commits: Record<string, string> };

export type FileKey = keyof typeof EVAL_FILES;
export type EvalData = Record<FileKey, unknown> & { commit_map?: CommitMap };
export type Ref = `${FileKey}:${string}`; // "summary:systems.agent.accuracy.value"

export const FORMATS = {
  pct: (v: number) => `${(v * 100).toFixed(1)}%`,
  pctRange: ([lo, hi]: [number, number]) => `${(lo * 100).toFixed(1)}–${(hi * 100).toFixed(1)}%`,
  pts: (v: number) => `${signed(v)} pts`,
  // test2 score values are stored as percent / points already (76.5, -0.5), not fractions.
  pctP: (v: number) => `${v.toFixed(1)}%`,
  pctPRange: ([lo, hi]: [number, number]) => `${lo.toFixed(1)}–${hi.toFixed(1)}%`,
  ptsP: (v: number) => `${signed(v / 100)} pts`,
  ptsPRange: ([lo, hi]: [number, number]) => `${signed2(lo)} to ${signed2(hi)}`,
  p2: (v: number) => v.toFixed(2),
  ptsRange: ([lo, hi]: [number, number]) => `${signed(lo)} to ${signed(hi)}`,
  secs: (ms: number) => `${(ms / 1000).toFixed(2)} s`,
  secsRange: ([lo, hi]: [number, number]) => `${(lo / 1000).toFixed(2)}–${(hi / 1000).toFixed(2)} s`,
  usd: (v: number) => `$${v.toFixed(4)}`,
  usdRange: ([lo, hi]: [number, number]) => `$${lo.toFixed(4)}–$${hi.toFixed(4)}`,
  frac: (count: number, n: number) => `${count}/${n}`,
  int: (v: number) => v.toLocaleString("en-US"),
  plain: (v: number) => String(v), // no thousands separator (resample count, as in the approved mockup)
  sum: (...vs: number[]) => vs.reduce((a, b) => a + b, 0).toLocaleString("en-US"),
  date: (...isos: string[]) => [...new Set(isos.map((s) => s.slice(0, 10)))].join(", "),
  commit: (sha: string) => sha.slice(0, 7),
  text: (s: string) => s,
} as const;
export type Format = keyof typeof FORMATS;

/** Points with 1 decimal; 2 when 1 would round a non-zero value to 0 and hide its sign. Unicode minus. */
export function signed(v: number): string {
  const p = v * 100;
  const text = p !== 0 && Math.abs(p) < 0.05 ? Math.abs(p).toFixed(2) : Math.abs(p).toFixed(1);
  return `${p < 0 ? "−" : "+"}${text}`;
}

/** Points already in points, 2 decimals (the scorer's bootstrap range). Unicode minus. */
export function signed2(p: number): string {
  return `${p < 0 ? "−" : "+"}${Math.abs(p).toFixed(2)}`;
}

export function resolve(data: EvalData, ref: Ref): unknown {
  const at = ref.indexOf(":");
  const file = ref.slice(0, at) as FileKey;
  let node: unknown = data[file];
  for (const part of ref.slice(at + 1).split(".")) {
    if (node === null || typeof node !== "object" || !(part in node)) throw new Error(`missing ${ref}`);
    node = (node as Record<string, unknown>)[part];
  }
  return node;
}

/** The recorded commit ID, with its current ID first when the commit map says it has changed. */
export function commitLabel(data: EvalData, recorded: string): string {
  const short = FORMATS.commit(recorded);
  if (!data.commit_map) return short;
  const current = data.commit_map.commits[recorded];
  if (!current) throw new Error(`commit map has no entry for ${recorded}`);
  return current === recorded ? short : `${FORMATS.commit(current)}, recorded as ${short}`;
}

export function formatRefs(data: EvalData, fmt: Format, refs: Ref[]): string {
  const values = refs.map((r) => resolve(data, r));
  if (fmt === "commit") return commitLabel(data, values[0] as string);
  return (FORMATS[fmt] as (...args: never[]) => string)(...(values as never[]));
}

/** Parses a data-src attribute back into refs ("a:x|b:y"). */
export const parseSrc = (src: string): Ref[] => src.split("|") as Ref[];
