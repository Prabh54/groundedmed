// Shared helpers for the browser tests. The fake API's trigger words (__weak__, __crash_retriever__ …) are
// documented at the top of scripts/mock-api.mjs.
import { expect, type Locator, type Page } from "@playwright/test";

export const API_URL = "http://localhost:8011"; // same value as playwright.config.ts

/** Types a question and presses Ask Becky. */
export async function ask(page: Page, question: string) {
  await page.locator("#q").fill(question);
  await page.getByRole("button", { name: "Ask Becky" }).click();
}

/** The page's red error box (Next.js adds its own empty role="alert" route announcer, so match the box). */
export const errorBox = (page: Page) => page.locator('.error-box[role="alert"]');

/** Waits for a full answer (the slim step row above "Becky's answer"). */
export async function waitForAnswer(page: Page) {
  await expect(page.getByRole("heading", { name: "Becky's answer" })).toBeVisible();
}

/** Records the `behavior` of every scrollIntoView call in window.__scrolls ("smooth" or "auto"). */
export async function recordScrolls(page: Page) {
  await page.addInitScript(() => {
    const w = window as unknown as { __scrolls: string[] };
    w.__scrolls = [];
    const original = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function (arg?: boolean | ScrollIntoViewOptions) {
      w.__scrolls.push(typeof arg === "object" && arg?.behavior ? arg.behavior : "auto");
      return original.call(this, arg);
    };
  });
}

export const scrolls = (page: Page) => page.evaluate(() => (window as unknown as { __scrolls: string[] }).__scrolls.slice());
export const clearScrolls = (page: Page) => page.evaluate(() => void ((window as unknown as { __scrolls: string[] }).__scrolls = []));

/** Records every distinct state of the working step list (".timeline") in window.__steps,
 *  one entry per change, e.g. "done,running,todo,todo". */
export async function recordSteps(page: Page) {
  await page.addInitScript(() => {
    const w = window as unknown as { __steps: string[] };
    w.__steps = [];
    const snap = () => {
      const items = [...document.querySelectorAll(".timeline li")];
      if (items.length !== 4) return;
      const state = items.map((li) => li.className.replace("step-", "")).join(",");
      if (w.__steps.at(-1) !== state) w.__steps.push(state);
    };
    new MutationObserver(snap).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
  });
}

export const steps = (page: Page) => page.evaluate(() => (window as unknown as { __steps: string[] }).__steps.slice());

/** Number of running glow animations on an element: Web Animations whose keyframes animate box-shadow
 *  (the card's border-colour CSS transition is not counted). */
export const glows = (locator: Locator) =>
  locator.evaluate((el) =>
    el.getAnimations().filter((a) => a.effect instanceof KeyframeEffect && a.effect.getKeyframes().some((k) => "boxShadow" in k)).length,
  );

/** Fails if the page can be scrolled sideways. */
export async function expectNoSidewaysScroll(page: Page) {
  const { scroll, client } = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    client: document.documentElement.clientWidth,
  }));
  expect(scroll, "page is wider than the screen").toBeLessThanOrEqual(client);
}
