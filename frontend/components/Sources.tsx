"use client";

import { anchorId, sectionsOf, type SourceView } from "@/lib/sources";

type Props = {
  views: SourceView[];
  open: Set<number>;
  marked: number | null; // last card jumped to from a [n]: blue border (the glow itself is animated by Ask.tsx)
  onToggle: (n: number) => void;
  countLine: string;
};

// Source cards from the approved mockup: number, PMID, PubMed link, what it supports, section pills,
// a 3-line preview, and the full conclusion-free passage when opened.
export default function Sources({ views, open, marked, onToggle, countLine }: Props) {
  return (
    <>
      <h3 className="sources-head">Sources</h3>
      <p className="sources-count">{countLine}</p>
      <ol className="source-list" aria-label="Sources">
        {views.map(({ n, pmid, citation, uses }) => {
          const isOpen = open.has(n);
          const textId = `${anchorId(n)}-text`;
          const sections = citation ? sectionsOf(citation) : [];
          return (
            <li key={n} id={anchorId(n)} tabIndex={-1} className={marked === n ? "source marked" : "source"}>
              <div className="source-top">
                <span className="num">
                  <span className="sr-only">Source </span>
                  {n}
                </span>
                <span className="pmid">PMID {pmid}</span>
                {citation && (
                  <a className="pubmed" href={citation.pubmed_url} target="_blank" rel="noopener noreferrer">
                    View on PubMed <span aria-hidden="true">↗</span>
                    <span className="sr-only"> (opens in a new tab)</span>
                  </a>
                )}
              </div>
              {uses && <p className="uses">{uses}</p>}
              {!citation ? (
                <p className="preview">Source text not available.</p>
              ) : (
                <>
                  {sections.some((s) => s.label) && (
                    <ul className="pills" aria-label="Sections">
                      {sections.filter((s) => s.label).map((s, i) => (
                        <li key={i}>{s.label}</li>
                      ))}
                    </ul>
                  )}
                  <div id={textId}>
                    {isOpen ? (
                      <div className="full">
                        {sections.map((s, i) => (
                          <p key={i}>
                            {s.label && (
                              <>
                                <b>{s.label}</b>
                                <br />
                              </>
                            )}
                            {s.text}
                          </p>
                        ))}
                        <p className="full-note">
                          This is the part of the abstract Becky read. The authors&apos; own conclusion is left out on purpose,
                          so Becky has to work from the study details. PubMed has the full abstract.
                        </p>
                      </div>
                    ) : (
                      <p className="preview" aria-hidden="true">
                        {sections.map((s, i) => (
                          <span key={i}>
                            {s.label && <b>{s.label}</b>} {s.text}{" "}
                          </span>
                        ))}
                      </p>
                    )}
                  </div>
                  <button className="toggle" type="button" aria-expanded={isOpen} aria-controls={textId} onClick={() => onToggle(n)}>
                    {isOpen ? "Show less" : "Show full text"}
                    <span className="sr-only"> of source {n}</span> <span aria-hidden="true">{isOpen ? "▴" : "▾"}</span>
                  </button>
                </>
              )}
            </li>
          );
        })}
      </ol>
    </>
  );
}
