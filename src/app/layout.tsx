import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vijay Tulluri | Data & AI Engineer",
  description: "Senior Software Engineer specializing in Data Engineering, Machine Learning, and Agentic AI solutions. Delivering enterprise-scale pipelines and AI innovations.",
  openGraph: {
    title: "Vijay Tulluri | Data & AI Engineer",
    description: "Senior Software Engineer specializing in Data Engineering, Machine Learning, and Agentic AI solutions.",
    url: "https://portfolio-vj.vercel.app", // Assuming this might be the URL or similar, can be updated later
    siteName: "Vijay Tulluri Portfolio",
    locale: "en_US",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
