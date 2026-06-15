import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenMatchER",
  description: "Open-source entity resolution and name matching platform"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

