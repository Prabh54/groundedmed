// Phone width (375 px, as the approved mockups): every state fits without sideways scrolling.
import { expect, test } from "@playwright/test";
import { ask, errorBox, expectNoSidewaysScroll, waitForAnswer } from "./helpers";

test.use({ viewport: { width: 375, height: 812 } });

test("home: fits, Ask button on the first screen, cards stacked, footer present", async ({ page }) => {
  await page.goto("/");
  await expectNoSidewaysScroll(page);
  const button = await page.getByRole("button", { name: "Ask Becky" }).boundingBox();
  expect(button!.y + button!.height).toBeLessThanOrEqual(812);
  const cards = await page.locator(".scope > div").evaluateAll((els) => els.map((el) => el.getBoundingClientRect().left));
  expect(new Set(cards.map(Math.round)).size).toBe(1); // one column
  await expect(page.locator(".site-footer")).toContainText("not medical advice");
  await expect(page.locator(".disclaimer")).toHaveCount(0); // the top bar is gone (owner, M5-T6)
});

test("answer fits", async ({ page }) => {
  await page.goto("/");
  await ask(page, "Does keyhole surgery help babies more?");
  await waitForAnswer(page);
  await page.getByRole("link", { name: "Source 1" }).first().click();
  await expect(page.locator("#source-1")).toBeFocused();
  await expectNoSidewaysScroll(page);
});

test("refusal fits", async ({ page }) => {
  await page.goto("/");
  await ask(page, "Does sky colour __weak__ change moods?");
  await expect(page.locator(".try-instead")).toBeVisible();
  await expectNoSidewaysScroll(page);
});

test("error fits", async ({ page }) => {
  await page.goto("/");
  await ask(page, "Does this step __crash_critic__ now?");
  await expect(errorBox(page)).toBeVisible();
  await expectNoSidewaysScroll(page);
});

test("/eval fits, gain cards stacked", async ({ page }) => {
  await page.goto("/eval");
  await expectNoSidewaysScroll(page);
  const lefts = await page.locator(".gains p").evaluateAll((els) => els.map((el) => Math.round(el.getBoundingClientRect().left)));
  expect(lefts[0]).toBe(lefts[1]);
});
