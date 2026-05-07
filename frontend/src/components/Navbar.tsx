"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const navLinks = [
    { href: "/", label: "Beranda" },
    { href: "/prediksi-cepat", label: "Prediksi" },
    { href: "/hasil", label: "Hasil" },
];

export default function Navbar() {
    const pathname = usePathname();

    return (
        <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2">
                </Link>

                {/* Links */}
                <div className="flex items-center gap-6">
                    {navLinks.map((link) => (
                        <Link
                            key={link.href}
                            href={link.href}
                            className={`text-sm font-medium transition-colors ${pathname === link.href
                                ? "text-indigo-600"
                                : "text-gray-500 hover:text-gray-900"
                                }`}
                        >
                            {link.label}
                        </Link>
                    ))}
                </div>
            </div>
        </nav>
    );
}