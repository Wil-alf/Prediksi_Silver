import Link from "next/link";
import Image from "next/image";
import { ArrowRight } from "lucide-react";

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

const author = {
  name: "Wilson Alfando",
  nim: "535220219",
  prodi: "Teknik Informatika",
  kampus: "Universitas Tarumanagara",
  image: "/author2.jpg",
};
const initials = author.name.split(" ").slice(0, 2).map((w) => w[0]).join("").toUpperCase();

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
              href="#cara-kerja"
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

      {/* Author Section */}
      <section className="bg-white py-14">
        <div className="max-w-6xl mx-auto px-6 flex flex-col items-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2 text-center">Tentang Penulis</h2>
          <p className="text-gray-500 text-sm mb-8 text-center">Penelitian ini merupakan bagian dari skripsi</p>
          <div className="bg-gray-50 border border-gray-100 rounded-2xl p-8 w-full max-w-lg flex flex-col sm:flex-row items-center gap-8">
            {/* Foto / Avatar */}
            <div className="w-36 h-36 rounded-2xl overflow-hidden shrink-0 bg-indigo-600 flex items-center justify-center">
              {author.image ? (
                <Image
                  src={author.image}
                  alt={author.name}
                  width={144}
                  height={144}
                  className="object-cover w-full h-full"
                />
              ) : (
                <span className="text-white text-4xl font-bold select-none">{initials}</span>
              )}
            </div>
            {/* Info */}
            <div className="flex flex-col gap-1 sm:items-start items-center text-center sm:text-left">
              <p className="text-xl font-semibold text-gray-800">{author.name}</p>
              <p className="text-sm text-gray-500">NIM: {author.nim}</p>
              <p className="text-sm text-gray-500">{author.prodi}</p>
              <p className="text-sm text-gray-500">{author.kampus}</p>
            </div>
          </div>
        </div>
      </section>

      {/* Cara Kerja */}
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
          <div className="text-center mt-12">
            <Link
              href="/prediksi-cepat"
              className="inline-flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition"
            >
              Mulai Sekarang <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
