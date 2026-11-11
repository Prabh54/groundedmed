// The live site from a clean browser: free checks first, then exactly 3 paid questions.
import { expect, test } from "@playwright/test";
import { EXAMPLES } from "../lib/examples";
import { ask, expectNoSidewaysScroll, waitForAnswer } from "../e2e/helpers";

const ANSWER_WAIT = { timeout: 120_000 }; // a real answer takes ~20-40 s

test("home page loads with its examples and footer", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Ask Becky" })).toBeVisible();
  await expect(page.getByRole("button", { name: EXAMPLES[0].question, exact: true })).toBeVisible();
  await expect(page.locator(".site-footer")).toHaveText(
    "GroundedMed is a portfolio research demo · not medical advice" +
      "Questions you ask are logged to run the demo, so please leave out personal details.",
  );
});

test("/eval loads its numbers and the note that the advice screen is outside the benchmark", async ({ page }) => {
  await page.goto("/eval");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("How GroundedMed is built and tested");
  await expect(page.getByText("Evaluation not run")).toHaveCount(0);
  expect(await page.locator("[data-src]").count()).toBeGreaterThan(100);
  await expect(page.locator("body")).toContainText("screens out questions asking for personal medical advice");
});

test.describe("phone width", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("home and /eval fit without sideways scrolling", async ({ page }) => {
    await page.goto("/");
    await expectNoSidewaysScroll(page);
    await page.goto("/eval");
    await expectNoSidewaysScroll(page);
  });
});

// --- paid: one question each ---------------------------------------------------------------

test("PAID: an example gets a full, cited answer and a citation opens its source", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: EXAMPLES[0].question, exact: true }).click();
  await page.getByRole("button", { name: "Ask Becky" }).click();

  await expect(page.getByRole("heading", { name: "Becky's answer" })).toBeVisible(ANSWER_WAIT);
  await waitForAnswer(page);
  await expect(page.locator(".verdict")).toHaveText(/^(Yes|No)$/);
  expect(await page.locator(".result ol.findings").first().locator("li").count()).toBeGreaterThan(0);
  await expect(page.locator(".trail .dot.done")).toHaveCount(4); // all four steps finished
  await expect(page.locator(`.source a[href*="${EXAMPLES[0].pmid}"]`).first()).toBeVisible(); // cites its own paper

  await page.getByRole("link", { name: "Source 1" }).first().click();
  const card = page.locator("#source-1");
  await expect(card).toBeFocused();
  await expect(card.getByRole("button", { name: /Show less/ })).toHaveAttribute("aria-expanded", "true");
  expect((await card.locator(".full").innerText()).length).toBeGreaterThan(200);
});

test("PAID: a casual wording with no close paper gets the rephrasing refusal", async ({ page }) => {
  await page.goto("/");
  await ask(page, "can teh brain right half side learn to read on its own");
  await expect(page.getByRole("heading", { name: "Not enough evidence" })).toBeVisible(ANSWER_WAIT);
  await expect(page.locator(".refusal")).toContainText("Try rephrasing with medical terms");
  await expect(page.locator(".error-box")).toHaveCount(0);
});

test("PAID: a personal-advice question gets the advice refusal with no steps", async ({ page }) => {
  await page.goto("/");
  await ask(page, "Is it ok for my dad to take tranexamic acid before his knee surgery?");
  await expect(page.getByRole("heading", { name: "Becky can't give medical advice" })).toBeVisible(ANSWER_WAIT);
  await expect(page.locator(".result")).not.toContainText("Try rephrasing");
  await expect(page.locator(".result .trail")).toHaveCount(0);
  await expect(page.locator(".error-box")).toHaveCount(0);
});
