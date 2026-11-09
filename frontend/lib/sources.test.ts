import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import type { Citation, Claim } from "./stream.ts";
import { anchorId, citeOrder, countText, sectionsOf, sourceViews, usesText } from "./sources.ts";

const cit = (pmid: string, sections = [{ label: "RESULTS", text: `text ${pmid}` }]): Citation => ({
  id: `PMID:${pmid}`, pmid, passage: `passage ${pmid}`, sections, pubmed_url: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`, rerank_score: 0.5,
});
const claim = (id: string, ...pmids: string[]): Claim => ({ id, text: id, citation_ids: pmids.map((p) => `PMID:${p}`) });

// Shape of the real API: `citations` = all 5 selected passages in evidence order, only some cited.
const RESULT = {
  claims: [claim("c1", "111"), claim("c2", "111", "222")],
  limitations: [claim("l1", "222")],
  citations: [cit("222"), cit("999"), cit("111"), cit("888"), cit("777")],
};

test("cards are numbered by first use in the answer, not by the API's evidence order", () => {
  assert.deepEqual(citeOrder(RESULT), ["PMID:111", "PMID:222"]);
  const views = sourceViews(RESULT);
  assert.deepEqual(views.map((v) => [v.n, v.pmid]), [[1, "111"], [2, "222"]]);
  assert.equal(views[0].citation?.passage, "passage 111");
  assert.equal(views[1].citation?.passage, "passage 222");
});

test("every citation in the answer has a card with its own passage and a unique anchor", () => {
  const views = sourceViews(RESULT);
  const cited = [...RESULT.claims, ...RESULT.limitations].flatMap((c) => c.citation_ids);
  for (const id of cited) {
    const v = views.find((x) => x.id === id);
    assert.ok(v, `${id} has a card`);
    assert.equal(v.citation?.id, id, "the card shows the passage the claim cited");
  }
  assert.equal(new Set(views.map((v) => anchorId(v.n))).size, views.length);
  assert.ok(!views.some((v) => ["PMID:999", "PMID:888", "PMID:777"].includes(v.id)), "uncited passages get no card");
});

test("'Supports' line names the findings and limitations each source backs, with no doubled 'and'", () => {
  const views = sourceViews(RESULT);
  assert.equal(views[0].uses, "Supports findings 1–2");
  assert.equal(views[1].uses, "Supports finding 2 and limitation 1");
  // the real answer of 2026-10-29 (2 findings + 3 limitations, all one source) read "findings 1 and 2 and limitations 1, 2 and 3"
  assert.equal(usesText([1, 2], [1, 2, 3]), "Supports findings 1–2 and limitations 1–3");
  assert.equal(usesText([1, 3], [2]), "Supports findings 1, 3 and limitation 2");
  assert.equal(usesText([1, 2, 3, 5, 7, 8], []), "Supports findings 1–3, 5, 7–8");
  assert.equal(usesText([], [1]), "Supports limitation 1");
  assert.equal(usesText([], []), "");
  for (const f of [[], [1], [1, 2], [1, 3], [1, 2, 4]]) {
    for (const l of [[], [1], [1, 2, 3], [2, 4]]) {
      assert.ok((usesText(f, l).match(/ and /g) ?? []).length <= 1, `at most one "and": ${usesText(f, l)}`);
    }
  }
});

test("count line uses the response's own counts", () => {
  assert.equal(countText(RESULT.citations.length, sourceViews(RESULT).length), "Becky read 5 abstracts and used 2 of them.");
  assert.equal(countText(5, 1), "Becky read 5 abstracts and used 1 of them.");
  assert.equal(countText(1, 1), "Becky read 1 abstract and used it.");
});

test("a cited ID the API did not send still gets a card, marked as having no text", () => {
  const views = sourceViews({ claims: [claim("c1", "123")], limitations: [], citations: [] });
  assert.deepEqual(views.map((v) => [v.n, v.pmid, v.citation]), [[1, "123", null]]);
});

test("no cited claims (weak evidence or no supported claims) means no cards", () => {
  assert.deepEqual(sourceViews({ claims: [], limitations: [], citations: RESULT.citations }), []);
});

// data/corpus.jsonl is not in Git (CI's frontend job has no copy), so this is a row in its exact shape:
// labels and text from PMID 26370095 (dev), shortened.
const CORPUS_ROW = {
  pmid: "26370095",
  sections: [
    { label: "AIMS", text: "To investigate the cost-effectiveness of up to £400 worth of financial incentives…" },
    { label: "DESIGN", text: "Cost-effectiveness analysis based on a Phase II randomized controlled trial…" },
    { label: "FINDINGS", text: "The incremental cost per quitter at 34-38 weeks pregnant was £1127…" },
  ],
};

test("labelled sections are shown as they come; plain passage is the fallback", () => {
  assert.deepEqual(sectionsOf(cit(CORPUS_ROW.pmid, CORPUS_ROW.sections)), CORPUS_ROW.sections);
  assert.deepEqual(sectionsOf(cit("1", [])), [{ label: "", text: "passage 1" }]);
});

test("a source cited twice in one claim is one card, numbered once", () => {
  const views = sourceViews({ claims: [claim("c1", "111", "111")], limitations: [], citations: [cit("111")] });
  assert.deepEqual(views.map((v) => [v.n, v.uses]), [[1, "Supports finding 1"]]);
});

test("the frontend Citation type matches the backend API model's fields", () => {
  const api = readFileSync(new URL("../../backend/app/models/api.py", import.meta.url), "utf8");
  const model = api.slice(api.indexOf("class Citation(BaseModel):"), api.indexOf("class Stage(BaseModel):"));
  const fields = [...model.matchAll(/^ {4}(\w+):/gm)].map((m) => m[1]).sort();
  assert.deepEqual(fields, Object.keys(cit("1")).sort());
});
