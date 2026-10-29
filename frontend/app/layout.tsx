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
        <div className="disclaimer" role="note">
          <strong>Research demo, not medical advice.</strong>
          <span className="disc-more"> Answers summarise published study abstracts. They are not guidance about anyone&apos;s health.</span>
        </div>
        {children}
      </body>
    </html>
  );
}
