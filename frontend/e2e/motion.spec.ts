// Motion with the browser set to "reduce motion", and the same checks with normal motion as the control,
// so a test that could never fail is caught: no bounce, no page flip, no placeholder rotation, no smooth
// scroll and no glow when reduced.
import { expect, test, type Page } from "@playwright/test";
import { EXAMPLES, ROTATE_MS } from "../lib/examples";
import { ask, clearScrolls, glows, recordScrolls, scrolls, waitForAnswer } from "./helpers";

const animationOf = (page: Page, selector: string) =>
  page.locator(selector).first().evaluate((el) => {
    const s = getComputedStyle(el);
    return { name: s.animationName, display: s.display };
  });

/** The placeholder after two rotation periods (fake clock), with the box empty and not focused. */
async function placeholderBeforeAfter(page: Page) {
  await page.clock.install();
  await page.goto("/");
  const box = page.locator("#q");
  // Proves the page is interactive (React loaded) before time is moved on; then empty and unfocused again.
  await page.getByRole("button", { name: EXAMPLES[0].question, exact: true }).click();
  await expect(box).toHaveValue(EXAMPLES[0].question);
  await box.fill("");
  await page.locator("h1").click();
  await expect(box).not.toBeFocused();
  const before = await box.getAttribute("placeholder");
  await page.clock.runFor(ROTATE_MS * 2 + 100);
  return { before, after: await box.getAttribute("placeholder") };
}

/** Asks, opens source 2 from its citation, and returns the scroll behaviours and the card's animation count. */
async function jumpToSource(page: Page) {
  await recordScrolls(page);
  await page.goto("/");
  await ask(page, "Does keyhole surgery help babies more?");
  await waitForAnswer(page);
  const askScrolls = await scrolls(page); // the "bring progress into view" scroll when the question starts
  await clearScrolls(page);
  await page.getByRole("link", { name: "Source 2" }).first().click();
  const card = page.locator("#source-2");
  await expect(card).toBeFocused();
  await expect(card).toBeInViewport();
  return { askScrolls, jumpScrolls: await scrolls(page), animations: await glows(card) };
}

test.describe("reduced motion", () => {
  test.use({ reducedMotion: "reduce" });

  test("Becky does not bounce", async ({ page }) => {
    await page.goto("/");
    expect((await animationOf(page, ".becky-bob")).name).toBe("none");
    expect((await animationOf(page, ".becky-floor")).name).toBe("none");
  });

  test("no page flip while Becky reads", async ({ page }) => {
    await page.goto("/");
    await ask(page, "Does the page flip __slow__ here?");
    await expect(page.locator(".becky-flip")).toBeAttached();
    expect(await animationOf(page, ".becky-flip")).toEqual({ name: "none", display: "none" });
    expect((await animationOf(page, ".becky-read")).name).toBe("none");
  });

  test("the placeholder does not rotate", async ({ page }) => {
    const { before, after } = await placeholderBeforeAfter(page);
    expect(after).toBe(before);
  });

  test("no smooth scrolling and no glow", async ({ page }) => {
    const { askScrolls, jumpScrolls, animations } = await jumpToSource(page);
    expect(askScrolls).toEqual(["auto"]);
    expect(jumpScrolls).toEqual(["auto"]);
    expect(animations).toBe(0);
    await expect(page.locator("#source-2")).toHaveClass(/marked/); // the blue border still shows which card
  });

  test("a suggested question after a refusal jumps to the box without smooth scrolling", async ({ page }) => {
    await recordScrolls(page);
    await page.goto("/");
    await ask(page, "Does sky colour __weak__ change moods?");
    await expect(page.locator(".try-instead")).toBeVisible();
    await clearScrolls(page);
    await page.locator(".try-instead .chip").first().click();
    await expect(page.locator("#q")).toBeFocused();
    expect(await scrolls(page)).toEqual(["auto"]);
  });
});

test.describe("normal motion (control)", () => {
  test.use({ reducedMotion: "no-preference" });

  test("Becky bounces", async ({ page }) => {
    await page.goto("/");
    expect((await animationOf(page, ".becky-bob")).name).toBe("becky-bob");
  });

  test("the page flips while Becky reads", async ({ page }) => {
    await page.goto("/");
    await ask(page, "Does the page flip __slow__ here?");
    await expect(page.locator(".becky-flip")).toBeAttached();
    expect((await animationOf(page, ".becky-flip")).name).toBe("becky-flip");
  });

  test("the placeholder rotates", async ({ page }) => {
    const { before, after } = await placeholderBeforeAfter(page);
    expect(after).not.toBe(before);
  });

  test("smooth scrolling and the glow", async ({ page }) => {
    const { askScrolls, jumpScrolls, animations } = await jumpToSource(page);
    expect(askScrolls).toEqual(["smooth"]);
    expect(jumpScrolls).toEqual(["smooth"]);
    expect(animations).toBe(1);
  });
});
