// Home page flows against the fake API: answer, refusals, citation opening, failures (M5-T6).
import { expect, test } from "@playwright/test";
import { BUTTON_COUNT, EXAMPLES } from "../lib/examples";
import { API_URL, ask, clearScrolls, errorBox, glows, recordScrolls, scrolls, waitForAnswer } from "./helpers";

test("an example question gets a cited answer with sources", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: EXAMPLES[0].question, exact: true }).click();
  await expect(page.locator("#q")).toHaveValue(EXAMPLES[0].question);
  await page.getByRole("button", { name: "Ask Becky" }).click();

  await waitForAnswer(page);
  await expect(page.getByRole("heading", { name: "Becky's answer" })).toBeFocused();
  await expect(page.locator(".verdict")).toHaveText("No");
  const lists = page.locator(".result ol.findings");
  await expect(lists.nth(0).locator("li")).toHaveCount(2); // findings
  await expect(page.getByRole("heading", { name: "Limitations" })).toBeVisible();
  await expect(lists.nth(1).locator("li")).toHaveCount(1); // limitations
  await expect(page.locator(".trail .dot.done")).toHaveCount(4);
  await expect(page.locator(".source")).toHaveCount(2); // a card per cited source; the other 3 only counted below
  await expect(page.locator(".sources-count")).toHaveText("Becky read 5 abstracts and used 2 of them.");
  await expect(page.locator(".try-instead")).toHaveCount(0);
  // no disclaimer note under the answer (owner, M5-T6); the site footer carries it
  await expect(page.locator(".result")).not.toContainText(/advice/i);
  await expect(page.locator(".site-footer")).toHaveText("GroundedMed is a portfolio research demo · not medical advice");
});

test("weak evidence: a friendly refusal with tips, and a tip fills the question box", async ({ page }) => {
  await recordScrolls(page);
  await page.goto("/");
  await ask(page, "Does sky colour __weak__ change moods?");

  const title = page.getByRole("heading", { name: "Not enough evidence" });
  await expect(title).toBeFocused();
  await expect(page.locator(".refusal")).toContainText(
    "Becky couldn't find a paper that closely matches this question, so she won't guess. Try rephrasing with medical " +
      "terms (e.g. 'right hemisphere' instead of 'right half of the brain'), or try one of the examples below.",
  );
  // not the wording of the other refusals
  await expect(page.locator(".refusal")).not.toContainText("couldn't back up an answer");
  await expect(page.getByRole("img", { name: "Becky shrugging with an apologetic smile" })).toBeVisible();
  await expect(page.locator(".error-box")).toHaveCount(0); // a refusal, not an error
  await expect(page.locator(".source")).toHaveCount(0);

  const tips = page.locator(".try-instead");
  await expect(tips.getByRole("heading", { name: "Try this instead" })).toBeVisible();
  await expect(tips).toContainText("Ask a yes / no question about what a study found.");
  await expect(tips).toContainText("Becky has only read 2,500 PubMed abstracts");
  const retry = tips.getByRole("group", { name: "Or try one she can answer:" }).getByRole("button");
  await expect(retry).toHaveText(EXAMPLES.slice(0, BUTTON_COUNT).map((e) => e.question));

  await clearScrolls(page);
  await retry.nth(1).click();
  const box = page.locator("#q");
  await expect(box).toHaveValue(EXAMPLES[1].question);
  await expect(box).toBeFocused();
  await expect(box).toBeInViewport();
  expect(await scrolls(page)).toEqual(["smooth"]);
  // filled, not sent: the refusal is still on screen and nothing new started
  await expect(title).toBeVisible();
  await expect(page.locator(".timeline")).toHaveCount(0);
});

test("personal-advice question: its own refusal, no steps, no rephrasing tip", async ({ page }) => {
  await page.goto("/");
  await ask(page, "Should I take this __advice__ for my condition?");
  const title = page.getByRole("heading", { name: "Becky can't give medical advice" });
  await expect(title).toBeFocused();
  const refusal = page.locator(".refusal");
  await expect(refusal).toContainText("Please ask a doctor or pharmacist.");
  await expect(page.locator(".result")).not.toContainText("Try rephrasing");
  await expect(page.locator(".result")).not.toContainText("Not enough evidence");
  await expect(page.locator(".result .trail")).toHaveCount(0); // nothing ran, so no steps
  await expect(page.locator(".source")).toHaveCount(0);
  await expect(page.locator(".error-box")).toHaveCount(0);
  const tips = page.locator(".try-instead");
  await expect(tips).toContainText("Ask what studies found");
  await expect(tips).not.toContainText("PubMed abstracts"); // the weak-evidence tips are not shown
  await expect(tips.getByRole("group", { name: "Or try one she can answer:" }).getByRole("button")).toHaveText(
    EXAMPLES.slice(0, BUTTON_COUNT).map((e) => e.question),
  );
});

test("model abstains but keeps a cited claim: refusal, tips and what the abstracts do say", async ({ page }) => {
  await page.goto("/");
  await ask(page, "Does this question __abstain__ please?");
  await expect(page.getByRole("heading", { name: "Not enough evidence" })).toBeVisible();
  await expect(page.locator(".refusal")).toContainText("don't answer this question directly");
  await expect(page.locator(".try-instead")).toBeVisible();
  await expect(page.getByRole("heading", { name: "What the abstracts do say" })).toBeVisible();
  await expect(page.locator(".result ol.findings li")).toHaveCount(1);
  await expect(page.locator(".source")).toHaveCount(1);
  await expect(page.locator(".sources-count")).toHaveText("Becky read 4 abstracts and used 1 of them.");
});

test("clicking a citation opens, scrolls to, focuses and highlights its source card", async ({ page }) => {
  await recordScrolls(page);
  await page.goto("/");
  await ask(page, "Does keyhole surgery help babies more?");
  await waitForAnswer(page);

  const card = page.locator("#source-2");
  await expect(card.getByRole("button", { name: /Show full text/ })).toHaveAttribute("aria-expanded", "false");
  await clearScrolls(page);
  await page.getByRole("link", { name: "Source 2" }).first().click();

  await expect(card).toBeFocused();
  await expect(card).toHaveClass(/marked/);
  await expect(card.getByRole("button", { name: /Show less/ })).toHaveAttribute("aria-expanded", "true");
  await expect(card.locator(".full")).toContainText("MOCK results for PMID");
  await expect(card).toBeInViewport();
  expect(await scrolls(page)).toEqual(["smooth"]);
  // the blue glow is a box-shadow Web Animation on the card (about 2 s)
  expect(await glows(card)).toBe(1);
});

test.describe("failures", () => {
  test("API unreachable", async ({ page }) => {
    await page.route(`${API_URL}/**`, (route) => route.abort());
    await page.goto("/");
    await ask(page, "Does keyhole surgery help babies more?");
    await expect(errorBox(page)).toHaveText("Can't reach the research service right now. Please try again later.");
    await expect(page.locator(".timeline")).toHaveCount(0); // nothing ran, so no step failed
  });

  test("service down before any step", async ({ page }) => {
    await page.goto("/");
    await ask(page, "Is the database __down__ now?");
    await expect(errorBox(page)).toHaveText("The service is temporarily unavailable, please try again later.");
    await expect(page.locator(".timeline")).toHaveCount(0);
  });

  test("demo limit reached (HTTP 429)", async ({ page }) => {
    await page.goto("/");
    await ask(page, "Am I over the __limit__ today?");
    await expect(errorBox(page)).toHaveText("Demo limit reached, please try again later.");
    await expect(page.locator(".timeline")).toHaveCount(0);
  });

  test("connection cut mid-answer", async ({ page }) => {
    await page.goto("/");
    await ask(page, "Will the stream __cut__ halfway?");
    await expect(errorBox(page)).toHaveText("Something went wrong while answering, please try again later.");
    await expect(page.locator(".timeline li.step-failed .label")).toHaveText("Synthesizing");
    await expect(page.getByRole("button", { name: "Ask Becky" })).toBeEnabled(); // can ask again
  });
});
