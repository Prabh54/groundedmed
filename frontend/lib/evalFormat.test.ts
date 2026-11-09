import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { loadEval } from "./evalData.ts";
import { FORMATS, commitLabel, formatRefs, parseSrc, resolve, signed, type EvalData } from "./evalFormat.ts";

test("formats: one decimal for percentages, 2 for seconds, 4 for dollars (expected values worked out by hand)", () => {
  assert.equal(FORMATS.pct(0.765), "76.5%");
  assert.equal(FORMATS.pct(0.7846153846153846), "78.5%");
  assert.equal(FORMATS.pct(1), "100.0%");
  assert.equal(FORMATS.pctRange([0.7015850701327535, 0.8184265467305365]), "70.2–81.8%");
  assert.equal(FORMATS.secs(11792.9), "11.79 s");
  assert.equal(FORMATS.secsRange([11482.65, 12082.6]), "11.48–12.08 s");
  assert.equal(FORMATS.usd(0.0287108132), "$0.0287");
  assert.equal(FORMATS.frac(153, 200), "153/200");
  assert.equal(FORMATS.int(1500), "1,500");
  assert.equal(FORMATS.sum(20, 10), "30");
  assert.equal(FORMATS.plain(2000), "2000");
  assert.equal(FORMATS.date("2026-10-25T19:46:39+00:00", "2026-10-25T20:06:18+00:00"), "2026-10-25");
  assert.equal(FORMATS.date("2026-10-25T23:59:00+00:00", "2026-10-26T00:01:00+00:00"), "2026-10-25, 2026-10-26");
  assert.equal(FORMATS.commit("498b76a615422e88def981d09fa98f6d7e328eff"), "498b76a");
});

test("signed points never hide a sign by rounding to zero", () => {
  assert.equal(signed(-0.025000000000000022), "−2.5");
  assert.equal(signed(0.225), "+22.5");
  assert.equal(signed(-0.0050000000000000044), "−0.5");
  assert.equal(signed(-0.00042807029314956003), "−0.04"); // claims strict lower end: "−0.0" would hide that it is below 0
  assert.equal(signed(0), "+0.0");
  assert.equal(FORMATS.ptsRange([-0.05, -0.005]), "−5.0 to −0.5");
});

test("refs resolve inside the stored files and fail loudly on a wrong path", () => {
  const data = loadEval();
  assert.ok(data, "the committed result files load");
  assert.equal(resolve(data, "summary:n_questions"), 200);
  assert.equal(formatRefs(data, "pct", ["summary:systems.baseline.accuracy.value"]), "76.5%");
  assert.equal(formatRefs(data, "frac", ["summary:systems.agent.accuracy.count", "summary:systems.agent.accuracy.n"]), "148/200");
  assert.throws(() => resolve(data, "summary:systems.baseline.accuracy.nope"), /missing/);
  assert.deepEqual(parseSrc("a:x|b:y"), ["a:x", "b:y"]);
});

test("missing result files mean 'evaluation not run', never placeholder numbers", () => {
  assert.equal(loadEval(mkdtempSync(path.join(tmpdir(), "no-eval-"))), null);
});

test("the shown results are the final test run, not a dev run", () => {
  const data = loadEval()!;
  assert.equal(resolve(data, "summary:split"), "test");
  assert.equal(resolve(data, "final_config:git_dirty"), false);
  assert.equal(resolve(data, "cross_config:name"), "second-AI cross-check");
  for (const s of ["baseline", "agent", "closed_book"]) assert.equal(resolve(data, `summary:systems.${s}.n_questions`), 200);
});

test("commit labels: recorded ID alone without a map; current ID first with one; a missing entry fails loudly", () => {
  const recorded = "498b76a615422e88def981d09fa98f6d7e328eff";
  const current = "c".repeat(40);
  const data = (commits?: Record<string, string>) => ({ commit_map: commits && { commits } }) as unknown as EvalData;
  assert.equal(commitLabel(data(), recorded), "498b76a");
  assert.equal(commitLabel(data({ [recorded]: current }), recorded), "ccccccc, recorded as 498b76a");
  assert.equal(commitLabel(data({ [recorded]: recorded }), recorded), "498b76a");
  assert.throws(() => commitLabel(data({}), recorded), /no entry/);
});
