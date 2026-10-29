import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Self-hosted so builds never download from Google Fonts (CI flaked on that, PLAN.md M6-F3).
// Latin variable files as served by Google Fonts (Fredoka v17, Inter v20); licences: app/fonts/*-OFL.txt.
const heading = localFont({ src: "./fonts/Fredoka-latin-variable.woff2", weight: "300 700", variable: "--font-heading" });
const body = localFont({ src: "./fonts/Inter-latin-variable.woff2", weight: "100 900", variable: "--font-body" });

export const metadata: Metadata = {
  title: "GroundedMed",
  description: "A research demo that answers from medical study abstracts and cites every finding. Not medical advice.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${heading.variable} ${body.variable}`}>
      <body>
        {children}
        {/* The site's disclaimer (owner, M5-T6: replaces the sticky top bar; the home page card repeats it). */}
        <footer className="site-footer">
          GroundedMed is a portfolio <strong>research demo</strong> · not medical advice
        </footer>
      </body>
    </html>
  );
}
