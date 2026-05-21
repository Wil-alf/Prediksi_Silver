import Link from "next/link";
import { ArrowRight, Download } from "lucide-react";

const steps = [
    {
        number: 1,
        title: "Pilih Periode Prediksi",
        desc: "Tentukan periode prediksi yang Anda inginkan, mulai dari 7 hari hingga 30 hari ke depan untuk melihat tren harga perak.",
    },
    {
        number: 2,
        title: "Data Historis",
        desc: "Sistem menampilkan grafik data historis harga perak sebagai referensi untuk memahami pola dan tren sebelumnya.",
    },
    {
        number: 3,
        title: "Proses Prediksi",
        desc: "Model machine learning menjalankan analisis menggunakan dua kombinasi algoritma hybrid untuk menghasilkan estimasi harga yang akurat.",
    },
    {
        number: 4,
        title: "Lihat Hasil Analisis",
        desc: "Bandingkan hasil prediksi dengan data aktual dalam grafik interaktif dan tabel metrik evaluasi untuk setiap algoritma.",
    },
];

export default function HelpPage() {
    return (
        <section id="cara-kerja" className="bg-gray-50 py-20">
            <div className="max-w-6xl mx-auto px-6">
                <h2 className="text-3xl font-bold text-gray-800 text-center mb-14">Cara Kerja Sistem</h2>
                <div className="grid gap-8">
                    {steps.map((step) => (
                        <div key={step.number} className="flex gap-6 items-start">
                            <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center font-bold text-sm shrink-0">
                                {step.number}
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-900 mb-1">{step.title}</h3>
                                <p className="text-gray-500 text-sm leading-relaxed">{step.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
                <div className="text-center mt-12 flex items-center justify-center gap-4 flex-wrap">
                    <Link
                        href="/prediksi-cepat"
                        className="inline-flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition"
                    >
                        Mulai Sekarang <ArrowRight size={16} />
                    </Link>
                    <a
                        href="Manual Book.pdf"
                        download
                        className="inline-flex items-center gap-2 border border-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition"
                    >
                        <Download size={16} /> Unduh Panduan Lengkap
                    </a>
                </div>
            </div>
        </section>
    );
}
