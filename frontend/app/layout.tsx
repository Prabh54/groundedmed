import type { Metadata } from "next";
import { Fredoka, Inter } from "next/font/google";
import "./globals.css";

const heading = Fredoka({ subsets: ["latin"], weight: ["500", "600"], variable: "--font-heading" });
const body = Inter({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-body" });

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
