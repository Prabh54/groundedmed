import Link from "next/link";
import Ask from "@/components/Ask";
import { CORPUS_ABSTRACTS } from "@/lib/examples";

export default function Home() {
  return (
    <main className="wrap">
      <header>
        <div className="logo">
          Grounded<span>Med</span>
        </div>
        <Link className="nav-pill" href="/eval">
          How well does it work? →
        </Link>
      </header>
      <Ask />
      <section className="scope" aria-label="What GroundedMed covers">
        <div>
          <h3>What she&apos;s read</h3>
          <p>{CORPUS_ABSTRACTS.toLocaleString("en-US")} PubMed study abstracts.</p>
        </div>
        <div>
          <h3>Best questions</h3>
          <p>Yes / no / maybe questions about a study.</p>
        </div>
        <div>
          <h3>No guessing</h3>
          <p>If the evidence is thin, Becky says so.</p>
        </div>
        <div className="cant">
          <h3>What she can&apos;t do</h3>
          <ul>
            <li>No new research: she only knows this fixed set of abstracts.</li>
            <li>She can be wrong: check the sources she links.</li>
            <li>Not medical advice: ask a doctor about your own health.</li>
          </ul>
          <Link href="/eval#h-limits">Full limitations →</Link>
        </div>
      </section>
    </main>
  );
}
