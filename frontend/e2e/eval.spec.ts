// /eval loads from the stored result files (exact values are proven by `npm run check:eval`; this checks the page in a browser).
import { expect, test } from "@playwright/test";

test("the /eval page loads with its numbers, the gain cards and its limitations", async ({ page }) => {
  await page.goto("/eval");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("How well does GroundedMed do?");
  await expect(page.getByText("Evaluation not run")).toHaveCount(0);
  expect(await page.locator("[data-src]").count()).toBeGreaterThan(100);
  for (const text of await page.locator("[data-src]").allTextContents()) expect(text.trim()).not.toBe("");

  // M5-T6: the two "vs closed-book" gains are large cards
  const gains = page.locator(".gains .gain");
  await expect(gains).toHaveCount(2);
  for (const gain of await gains.all()) {
    await expect(gain).toHaveText(/^[+−]\d+\.\d pts$/);
    const size = await gain.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
    expect(size).toBeGreaterThanOrEqual(44);
  }
  await expect(page.locator(".gains p").nth(0)).toContainText("baseline vs closed-book");
  await expect(page.locator(".gains p").nth(1)).toContainText("agent vs closed-book");

  await expect(page.getByRole("heading", { name: "Limitations" })).toBeVisible();
  await expect(page.locator(".site-footer")).toContainText("not medical advice");
});

test("the home page's 'Full limitations' link opens the limitations on /eval", async ({ page }) => {
  await page.goto("/");
  const card = page.locator(".scope .cant");
  await expect(card.getByRole("heading", { name: "What she can't do" })).toBeVisible();
  await expect(card).toContainText("Not medical advice");
  await card.getByRole("link", { name: "Full limitations →" }).click();
  await expect(page).toHaveURL(/\/eval#h-limits$/);
  await expect(page.locator("#h-limits")).toBeInViewport();
});
