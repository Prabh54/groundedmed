// Example questions: short rewordings of DEV questions (never test), each run once through the agent
// (2026-10-28; the twins and breast-cancer ones 2026-10-30 UTC) and kept only if it gave a cited yes/no answer
// matching the dev label (docs/design/example-questions-check.json). lib/examples.test.ts checks this list against that file and
// the frozen manifests. The first BUTTON_COUNT are also the buttons; all of them rotate in the placeholder.
// Removed by the owner as treatment-flavoured: rt-PA stroke drug (24669960, 2026-10-29) and keyhole vs open
// surgery in newborns (21194998, 2026-10-30 UTC). Candidate 11138995 answered "maybe" in its one check: not added.
// Brain's right half (25819796) replaced 2026-11-02 by the nursing-homes question: it cleared the evidence gate by only
// 0.03, so casual retypes were refused. The replacement was picked by a rule fixed in advance (PROGRESS session 16
// part 5): gate margin >= 0.10 for this wording and a casual retype still passing.
export const EXAMPLES: { question: string; pmid: string }[] = [
  { question: "Can third-trimester ultrasound predict the position of the first twin at delivery?", pmid: "22680064" },
  { question: "Are people with serious mental illness more likely to end up in worse nursing homes?", pmid: "21368683" },
  { question: "Do recorded lectures make medical students skip class?", pmid: "28027677" },
  { question: "Is paying pregnant smokers to quit worth the cost?", pmid: "26370095" },
  { question: "Does a baby's sex affect asthma ER visits during pregnancy?", pmid: "16809243" },
  { question: "Can adults with mild intellectual disabilities get better at reading comprehension?", pmid: "17910536" },
  { question: "Is breast cancer prognosis inherited?", pmid: "17598882" },
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
