// Source cards (approved mockup docs/design/m5-sources-mockup.html): pure logic, unit-tested.
// `citations` from the API are all the evidence passages Becky selected; cards are only the cited ones.

import type { Citation, Claim, QueryResponse } from "./stream.ts";

export type SourceView = {
  n: number; // citation number shown as [n]
  id: string; // "PMID:123"
  pmid: string;
  citation: Citation | null; // null only if the API cited an ID it did not send (never expected)
  uses: string; // "Supports findings 1 and 2"
};

export const anchorId = (n: number) => `source-${n}`;

/** Cited IDs in order of first use, claims before limitations, so [1] is the first source the answer uses. */
export function citeOrder(result: Pick<QueryResponse, "claims" | "limitations">): string[] {
  const order: string[] = [];
  for (const item of [...result.claims, ...result.limitations]) {
    for (const id of item.citation_ids) if (!order.includes(id)) order.push(id);
  }
  return order;
}

/** Runs in a row become ranges, the rest commas: [1, 2, 3, 5] -> "1–3, 5". No "and", so it never doubles up. */
export function joinNumbers(ns: number[]): string {
  const parts: string[] = [];
  for (let i = 0; i < ns.length; i++) {
    let j = i;
    while (j + 1 < ns.length && ns[j + 1] === ns[j] + 1) j++;
    parts.push(j > i ? `${ns[i]}–${ns[j]}` : `${ns[i]}`);
    i = j;
  }
  return parts.join(", ");
}

/** "Supports finding 2", "Supports findings 1–2", "Supports findings 1–2 and limitations 1–3". */
export function usesText(findings: number[], limitations: number[]): string {
  const parts: string[] = [];
  if (findings.length) parts.push(`${findings.length === 1 ? "finding" : "findings"} ${joinNumbers(findings)}`);
  if (limitations.length) parts.push(`${limitations.length === 1 ? "limitation" : "limitations"} ${joinNumbers(limitations)}`);
  return parts.length ? `Supports ${parts.join(" and ")}` : "";
}

function positionsCiting(items: Claim[], id: string): number[] {
  return items.flatMap((item, i) => (item.citation_ids.includes(id) ? [i + 1] : []));
}

export function sourceViews(result: Pick<QueryResponse, "claims" | "limitations" | "citations">): SourceView[] {
  const byId = new Map(result.citations.map((c) => [c.id, c]));
  return citeOrder(result).map((id, i) => {
    const citation = byId.get(id) ?? null;
    return {
      n: i + 1,
      id,
      pmid: citation?.pmid ?? id.replace(/^PMID:/, ""),
      citation,
      uses: usesText(positionsCiting(result.claims, id), positionsCiting(result.limitations, id)),
    };
  });
}

/** "Becky read 5 abstracts and used 2 of them." Counts come from the API response, never fixed. */
export function countText(read: number, used: number): string {
  if (read === 1 && used === 1) return "Becky read 1 abstract and used it.";
  return `Becky read ${read} ${read === 1 ? "abstract" : "abstracts"} and used ${used} of them.`;
}

/** Labelled sections to show; falls back to the plain passage if a document has no section labels. */
export function sectionsOf(c: Citation): { label: string; text: string }[] {
  return c.sections.length ? c.sections : [{ label: "", text: c.passage }];
}
