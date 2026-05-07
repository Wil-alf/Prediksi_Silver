import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Peramalan Perak",
  description: "Prediksi harga perak dengan Prophet+XGBoost dan Prophet+LightGBM",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="id">
      <body className={`${inter.className} bg-gray-50 text-gray-900`}>
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}