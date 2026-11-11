import type { Metadata } from "next";
import Link from "next/link";
import Becky from "@/components/Becky";
import EvalReport from "@/components/EvalReport";
import { loadEval } from "@/lib/evalData";

export const metadata: Metadata = {
  title: "How it's built and tested · GroundedMed",
  description: "How GroundedMed's agentic RAG pipeline works and how it was tested on two frozen PubMedQA test sets, with ranges and limitations.",
};

// Static: the stored result files are read once at build time, never on page load.
export const dynamic = "force-static";

export default function EvalPage() {
  const data = loadEval();
  // On CI or Vercel the files must be there: never ship a silent "evaluation not run" by mistake.
  if (!data && (process.env.CI || process.env.VERCEL)) {
    throw new Error("Stored evaluation result files not found at build time (lib/evalFormat.ts EVAL_FILES).");
  }
  return (
    <main className="wrap eval">
      <header>
        <Link href="/" className="logo">
          Grounded<span>Med</span>
        </Link>
        <span className="becky-small" aria-hidden="true">
          <Becky pose="normal" width={44} />
        </span>
        <Link className="nav-pill" href="/">
          ← Ask Becky
        </Link>
      </header>
      {data ? (
        <EvalReport data={data} />
      ) : (
        <section className="eval-card">
          <h1>Evaluation not run</h1>
          <p>The stored evaluation results were not found when this page was built, so no numbers are shown.</p>
        </section>
      )}
    </main>
  );
}
