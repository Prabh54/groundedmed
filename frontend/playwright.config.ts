// Browser (E2E) tests. $0: the app talks only to the fake API (scripts/mock-api.mjs), never the real one.
// Own ports and build folder, so a dev server on :3000 / mock on :8001 can keep running alongside.
// Run: npm run test:e2e   (first time on a new machine: npx playwright install chromium)
import { defineConfig, devices } from "@playwright/test";

const API_PORT = 8011;
const APP_PORT = 3011;
const API_URL = `http://localhost:${API_PORT}`; // e2e/helpers.ts has the same value

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: 0, // a flaky browser test should fail loudly, not pass on a second try
  workers: process.env.CI ? 2 : undefined,
  reporter: "list",
  // Normal motion unless a test asks for reduced motion (motion.spec.ts), whatever the machine's own setting.
  use: { baseURL: `http://localhost:${APP_PORT}`, trace: "retain-on-failure", reducedMotion: "no-preference" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "node scripts/mock-api.mjs",
      url: `${API_URL}/health`,
      env: { MOCK_PORT: String(API_PORT), MOCK_ORIGINS: `http://localhost:${APP_PORT}` },
      reuseExistingServer: false,
    },
    {
      // The production build, as visitors get it (not the dev server).
      command: `npx next build && npx next start -p ${APP_PORT}`,
      url: `http://localhost:${APP_PORT}`,
      env: { NEXT_PUBLIC_API_BASE_URL: API_URL, NEXT_DIST_DIR: ".next-e2e", NEXT_TELEMETRY_DISABLED: "1" },
      timeout: 240_000,
      reuseExistingServer: false,
    },
  ],
});
