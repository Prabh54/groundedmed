// Example questions: short rewordings of DEV questions (never test), each run once through the agent on
// 2026-10-28 and kept only if it gave a cited yes/no answer matching the dev label
// (docs/design/example-questions-check.json). lib/examples.test.ts checks this list against that file and
// the frozen manifests. The first BUTTON_COUNT are also the buttons; all of them rotate in the placeholder.
// The rt-PA stroke-drug example (24669960) was removed by the owner on 2026-10-29: it read closest to treatment
// advice. Its replacement candidate (11138995) answered "maybe" in its one check and was not added.
export const EXAMPLES: { question: string; pmid: string }[] = [
  { question: "Does keyhole surgery hurt newborns less than open surgery?", pmid: "21194998" },
  { question: "Can the brain's right half learn to read on its own?", pmid: "25819796" },
  { question: "Do recorded lectures make medical students skip class?", pmid: "28027677" },
  { question: "Is paying pregnant smokers to quit worth the cost?", pmid: "26370095" },
  { question: "Does a baby's sex affect asthma ER visits during pregnancy?", pmid: "16809243" },
  { question: "Can adults with mild intellectual disabilities get better at reading comprehension?", pmid: "17910536" },
];

export const BUTTON_COUNT = 3;
export const ROTATE_MS = 4000;

// Number of abstracts in the frozen corpus (data/manifests/corpus.json; checked by the test).
export const CORPUS_ABSTRACTS = 2500;

// Next placeholder example: stays put while the visitor is in the box, has typed, or prefers reduced motion.
export function nextExample(index: number, paused: { focused: boolean; hasText: boolean; reducedMotion: boolean }): number {
  if (paused.focused || paused.hasText || paused.reducedMotion) return index;
  return (index + 1) % EXAMPLES.length;
}
