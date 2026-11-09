// The four-step list drawn in a real browser: ticks in order; a failing step shows ✕ and later ones "not started".
import { expect, test } from "@playwright/test";
import { STEP_LABELS } from "../lib/timeline";
import { ask, errorBox, recordSteps, steps, waitForAnswer } from "./helpers";

const RANK: Record<string, number> = { todo: 0, running: 1, done: 2 };

test("the four steps tick in order", async ({ page }) => {
  await recordSteps(page);
  await page.goto("/");
  await ask(page, "Does keyhole surgery help babies more?");
  await expect(page.locator(".timeline li .label")).toHaveText([...STEP_LABELS]);
  await waitForAnswer(page);

  const seen = (await steps(page)).map((s) => s.split(","));
  // Every step was seen running while all earlier steps were done and all later ones not yet started.
  STEP_LABELS.forEach((label, i) => {
    const expected = STEP_LABELS.map((_, j) => (j < i ? "done" : j === i ? "running" : "todo")).join(",");
    expect(seen.map((s) => s.join(",")), `${label} running in order`).toContain(expected);
  });
  // No step ever went backwards (todo -> running -> done only), and nothing failed or was skipped.
  for (let k = 1; k < seen.length; k++) {
    seen[k].forEach((status, j) => {
      expect(RANK[status], `step ${STEP_LABELS[j]} in state ${k}`).toBeGreaterThanOrEqual(RANK[seen[k - 1][j]]);
    });
  }
  // After the answer: the slim row shows four ticks.
  await expect(page.locator(".trail li .label")).toHaveText([...STEP_LABELS]);
  await expect(page.locator(".trail .dot")).toHaveText(["✓", "✓", "✓", "✓"]);
});

const FAILING = ["planner", "retriever", "synthesizer", "critic"] as const;

FAILING.forEach((node, k) => {
  test(`a failing ${STEP_LABELS[k]} step shows ✕ and the later steps "not started"`, async ({ page }) => {
    await page.goto("/");
    await ask(page, `Does this step __crash_${node}__ now?`);
    await expect(errorBox(page)).toBeVisible();
    const items = page.locator(".timeline li");
    await expect(items).toHaveCount(4);
    for (let j = 0; j < 4; j++) {
      const item = items.nth(j);
      if (j < k) {
        await expect(item).toHaveClass("step-done");
        await expect(item.locator(".dot")).toHaveText("✓");
      } else if (j === k) {
        await expect(item).toHaveClass("step-failed");
        await expect(item.locator(".dot")).toHaveText("✕");
        await expect(item.locator(".meta")).toHaveText(" · didn't finish");
      } else {
        await expect(item).toHaveClass("step-todo");
        await expect(item.locator(".meta")).toHaveText(" · not started");
      }
    }
    await expect(page.getByRole("img", { name: "Becky shrugging with an apologetic smile" })).toBeVisible();
  });
});
