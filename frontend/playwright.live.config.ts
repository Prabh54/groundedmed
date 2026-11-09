// Browser tests against the LIVE site. MANUAL ONLY and PAID: 3 real questions per run (~$0.035).
// Never run by CI or on push: CI's `npm run test:e2e` uses playwright.config.ts (testDir e2e), and this file
// refuses to load unless LIVE_E2E=1 (a backend test checks both). Run: npm run test:e2e:live
// Each test gets a fresh browser context (no cookies or storage), as a first-time visitor would.
import { defineConfig, devices } from "@playwright/test";

if (process.env.LIVE_E2E !== "1") {
  throw new Error("Live browser tests send paid questions: run them only with `npm run test:e2e:live`.");
}

export default defineConfig({
  testDir: "e2e-live",
  timeout: 180_000,
  expect: { timeout: 15_000 },
  retries: 0, // a retry would send another paid question
  workers: 1,
  fullyParallel: false,
  reporter: "list",
  outputDir: "test-results-live",
  use: { baseURL: "https://groundedmed.vercel.app", trace: "off", screenshot: "on" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
