import type { NextConfig } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants";

export default function config(phase: string): NextConfig {
  // A production build without the API URL would silently call localhost from visitors' browsers.
  if (phase === PHASE_PRODUCTION_BUILD && !process.env.NEXT_PUBLIC_API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL must be set for a production build.");
  }
  // The Playwright run builds into its own folder so it never overwrites a dev or deploy build (playwright.config.ts).
  return { reactStrictMode: true, distDir: process.env.NEXT_DIST_DIR || ".next" };
}
