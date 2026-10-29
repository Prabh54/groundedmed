// Build-time loader for the /eval page (ADR-016): reads the committed result files from the repo.
// Returns null when any file is missing, so the page says "evaluation not run" instead of guessing.

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { EVAL_FILES, type EvalData, type FileKey } from "./evalFormat.ts";

// `next build` and the tests both run from frontend/; the result files live one level up.
export const REPO_ROOT = path.resolve(process.cwd(), "..");

export function loadEval(root: string = REPO_ROOT): EvalData | null {
  const data: Partial<EvalData> = {};
  for (const [key, rel] of Object.entries(EVAL_FILES) as [FileKey, string][]) {
    const file = path.join(root, rel);
    if (!existsSync(file)) return null;
    data[key] = JSON.parse(readFileSync(file, "utf8"));
  }
  return data as EvalData;
}
