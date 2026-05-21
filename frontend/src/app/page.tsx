import Link from "next/link";
import { ArrowRight } from "lucide-react";

const stats = [
  { label: "Algoritma Hybrid", value: "2 Model" },
  { label: "Data Historis", value: "10 Tahun" },
  { label: "Periode Maksimum", value: "30 Hari" },
  { label: "Metrik Evaluasi", value: "4 Jenis" },
];

const variables = [
  { ticker: "SI=F", name: "Harga Perak", color: "bg-indigo-100 text-indigo-700" },
  { ticker: "GC=F", name: "Harga Emas", color: "bg-yellow-100 text-yellow-700" },
  { ticker: "CL=F", name: "Harga Minyak", color: "bg-orange-100 text-orange-700" },
  { ticker: "DX-Y.NYB", name: "Nilai Tukar USD", color: "bg-green-100 text-green-700" },
];

export default function BerandaPage() {
  return (
    <>
      {/* Hero Section */}
      <section className="max-w-6xl mx-auto px-6 py-20 flex items-center gap-16 bg-white text-slate-900">
        <div className="flex-1">
          <h1 className="text-5xl font-bold text-gray-900 leading-tight mb-6">
            Peramalan <br />
            Harga <span className="text-gray-800">Perak</span>
          </h1>
          <p className="text-gray-500 text-lg mb-8 max-w-md">
            Alat prediksi harga perak menggunakan teknologi machine learning
            dengan perbandingan algoritma Prophet+XGBoost dan Prophet+LightGBM
            berbasis data multivariat.
          </p>
          <div className="flex gap-4">
            <Link
              href="/prediksi-cepat"
              className="inline-flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition"
            >
              Mulai Prediksi <ArrowRight size={16} />
            </Link>
            <Link
              href="/help"
              className="inline-flex items-center gap-2 border border-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition"
            >
              Pelajari Lebih Lanjut
            </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="flex-1 hidden md:grid grid-cols-2 gap-4">
          {stats.map(({ label, value }) => (
            <div key={label} className="bg-indigo-50 rounded-2xl p-6 flex flex-col gap-1">
              <span className="text-xs text-indigo-400 font-medium">{label}</span>
              <span className="text-2xl font-bold text-indigo-700">{value}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Variables Section */}
      <section className="bg-gray-50 py-14">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Variabel yang Digunakan</h2>
          <p className="text-gray-500 text-sm mb-8">Data diambil secara otomatis dari Yahoo Finance</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {variables.map(({ ticker, name, color }) => (
              <div key={ticker} className="bg-white rounded-2xl border border-gray-100 p-5">
                <span className={`inline-block text-xs font-mono font-semibold px-2 py-1 rounded-md mb-3 ${color}`}>
                  {ticker}
                </span>
                <p className="text-sm font-medium text-gray-800">{name}</p>
                <p className="text-xs text-gray-400 mt-1">Harga penutupan harian</p>
              </div>
            ))}
          </div>
        </div>
      </section>

    </>
  );
}
