import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { BUTTON_COUNT, CORPUS_ABSTRACTS, EXAMPLES, nextExample } from "./examples.ts";

const repo = new URL("../../", import.meta.url);
const readJson = (path: string) => JSON.parse(readFileSync(new URL(path, repo), "utf8"));
const items = (split: string): { pmid: string; final_decision: string }[] => readJson(`data/manifests/${split}.json`).items;
const pmids = (split: string) => new Set(items(split).map((i) => i.pmid));

test("examples come from dev papers and never from the frozen test set", () => {
  const dev = pmids("dev");
  const test_ = pmids("test");
  for (const e of EXAMPLES) {
    assert.ok(dev.has(e.pmid), `${e.pmid} is not a dev paper`);
    assert.ok(!test_.has(e.pmid), `${e.pmid} is a TEST paper`);
  }
});

test("every example passed the recorded agent check: cited yes/no answer matching the dev label", () => {
  const devLabel = new Map(items("dev").map((i) => [i.pmid, i.final_decision]));
  const checked = new Map<string, Record<string, unknown>>(
    readJson("docs/design/example-questions-check.json").results.map((r: { pmid: string }) => [r.pmid, r]),
  );
  for (const e of EXAMPLES) {
    const r = checked.get(e.pmid);
    assert.ok(r, `${e.pmid} was never checked`);
    assert.equal(r.question, e.question, "the page must show the exact wording that was checked");
    assert.equal(r.abstained, false);
    assert.equal(r.keep, true);
    assert.ok(r.verdict === "yes" || r.verdict === "no");
    assert.equal(r.verdict, devLabel.get(e.pmid), "verdict must equal the frozen dev manifest's label");
    assert.ok((r.cited as string[]).includes(`PMID:${e.pmid}`), "answer cites its own source paper");
  }
});

// PLAN.md's "3-6 examples" are the clickable buttons; the placeholder also rotates through the other checked ones.
test("3 to 6 example buttons, no duplicate examples, and the scope card's abstract count matches the corpus", () => {
  assert.ok(BUTTON_COUNT >= 3 && BUTTON_COUNT <= 6 && BUTTON_COUNT <= EXAMPLES.length);
  assert.equal(new Set(EXAMPLES.map((e) => e.question)).size, EXAMPLES.length);
  assert.equal(new Set(EXAMPLES.map((e) => e.pmid)).size, EXAMPLES.length);
  const corpus = readJson("data/manifests/corpus.json");
  assert.equal(corpus.count, CORPUS_ABSTRACTS);
  assert.equal(corpus.documents.length, CORPUS_ABSTRACTS);
});

test("placeholder rotation advances, wraps, and pauses on focus, typed text or reduced motion", () => {
  const free = { focused: false, hasText: false, reducedMotion: false };
  assert.equal(nextExample(0, free), 1);
  assert.equal(nextExample(EXAMPLES.length - 1, free), 0);
  assert.equal(nextExample(2, { ...free, focused: true }), 2);
  assert.equal(nextExample(2, { ...free, hasText: true }), 2);
  assert.equal(nextExample(2, { ...free, reducedMotion: true }), 2);
});
