// M5-T5 acceptance check, run after `npm run build` (CI does): see lib/checkEval.ts for what it proves.
// Run: node scripts/check-eval-page.ts [path/to/eval.html]   (default: the Next.js build output)

import { readFileSync } from "node:fs";
import { checkEvalHtml } from "../lib/checkEval.ts";
import { loadEval } from "../lib/evalData.ts";

const htmlFile = process.argv[2] ?? ".next/server/app/eval.html";
const data = loadEval();
if (!data) {
  console.error("check-eval-page: result files missing: the page would say 'evaluation not run'");
  process.exit(1);
}
const { count, problems } = checkEvalHtml(readFileSync(htmlFile, "utf8"), data);
if (problems.length) {
  console.error(`check-eval-page: ${problems.length} problem(s) in ${htmlFile}\n- ${problems.join("\n- ")}`);
  process.exit(1);
}
console.log(`check-eval-page: ${count} numbers on ${htmlFile} match the stored result files; no unsourced numbers; wording ok`);
