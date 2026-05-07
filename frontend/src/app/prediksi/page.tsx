"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Play } from "lucide-react";
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import { runPrediction, getHistoricalData, type HistoricalRow } from "@/lib/api";

function formatDate(d: Date): string {
    return d.toISOString().slice(0, 10);
}

const TODAY    = formatDate(new Date());
const MIN_DATE = (() => { const d = new Date(); d.setFullYear(d.getFullYear() - 10); return formatDate(d); })();

export default function PrediksiPage() {
    const [period, setPeriod]           = useState<7 | 30>(7);
    const [startDate, setStartDate]     = useState(MIN_DATE);
    const [endDate, setEndDate]         = useState(TODAY);
    const [loading, setLoading]         = useState(false);
    const [historical, setHistorical]   = useState<HistoricalRow[]>([]);
    const [histLoading, setHistLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        if (!startDate || !endDate || startDate >= endDate) return;
        const timer = setTimeout(() => {
            setHistLoading(true);
            getHistoricalData(startDate, endDate)
                .then(setHistorical)
                .catch(console.error)
                .finally(() => setHistLoading(false));
        }, 600);
        return () => clearTimeout(timer);
    }, [startDate, endDate]);

    const handleRunPrediction = async () => {
        setLoading(true);
        try {
            const result = await runPrediction(period, endDate || undefined);
            sessionStorage.setItem("predictionResult", JSON.stringify(result));
            router.push("/hasil");
        } catch {
            alert("Gagal menjalankan prediksi. Pastikan backend aktif.");
        } finally {
            setLoading(false);
        }
    };

    const dateRangeInvalid = !startDate || !endDate || startDate >= endDate;
    const tickInterval     = historical.length > 0 ? Math.floor(historical.length / 6) : 30;

    return (
        <div className="max-w-6xl mx-auto px-6 py-10 bg-gray-50">
            <Link href="/" className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-900 mb-6">
                <ChevronLeft size={16} /> Kembali ke Beranda
            </Link>

            <h1 className="text-3xl font-bold mb-1 text-gray-800">Halaman Prediksi</h1>
            <p className="text-gray-500 mb-8">
                Pilih periode prediksi dan jalankan analisis untuk mendapatkan estimasi harga perak.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Parameter Card */}
                <div className="bg-white rounded-2xl border border-gray-200 p-6">
                    <h2 className="font-semibold text-lg mb-4 text-gray-800">Parameter Prediksi</h2>

                    <p className="text-sm text-gray-500 mb-3">Periode Prediksi</p>
                    <div className="flex gap-2 mb-6">
                        {([7, 30] as const).map((p) => (
                            <button
                                key={p}
                                onClick={() => setPeriod(p)}
                                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition ${period === p
                                    ? "bg-indigo-600 border-indigo-600 text-white"
                                    : "border-gray-200 text-gray-600 hover:border-indigo-300"
                                    }`}
                            >
                                {p} Hari
                            </button>
                        ))}
                    </div>

                    <p className="text-sm text-gray-500 mb-3">Variabel Multivariat</p>
                    <div className="flex flex-col gap-1.5 mb-6">
                        {[
                            { label: "Harga Perak (SI=F)",          color: "bg-indigo-500" },
                            { label: "Harga Emas (GC=F)",           color: "bg-yellow-500" },
                            { label: "Harga Minyak (CL=F)",         color: "bg-orange-500" },
                            { label: "Nilai Tukar USD (DX-Y.NYB)",  color: "bg-green-500" },
                        ].map(({ label, color }) => (
                            <div key={label} className="flex items-center gap-2 text-sm text-gray-700">
                                <span className={`w-2 h-2 rounded-full ${color} inline-block`} />
                                {label}
                            </div>
                        ))}
                    </div>

                    <p className="text-sm text-gray-500 mb-3">Algoritma yang Digunakan</p>
                    <div className="flex flex-col gap-2 mb-6">
                        {["Prophet + XGBoost", "Prophet + LightGBM"].map((algo) => (
                            <div key={algo} className="flex items-center gap-2 text-sm text-gray-700">
                                <span className="w-2 h-2 rounded-full bg-indigo-600 inline-block" />
                                {algo}
                            </div>
                        ))}
                    </div>

                    <p className="text-sm text-gray-500 mb-3">Train Test split</p>
                    <div className="flex flex-col gap-2 mb-6">
                        {["80% data train", "20% data testing"].map((algo) => (
                            <div key={algo} className="flex items-center gap-2 text-sm text-gray-700">
                                <span className="w-2 h-2 rounded-full bg-indigo-600 inline-block" />
                                {algo}
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={handleRunPrediction}
                        disabled={loading}
                        className="w-full inline-flex items-center justify-center gap-2 bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 transition disabled:opacity-60"
                    >
                        <Play size={16} />
                        {loading ? "Memproses..." : "Jalankan Prediksi"}
                    </button>
                </div>

                {/* Historical Chart */}
                <div className="bg-white rounded-2xl border border-gray-200 p-6">
                    <h2 className="font-semibold text-lg mb-1 text-gray-800">Data Historis Harga Perak</h2>
                    <p className="text-xs text-gray-500 mb-3">Pratinjau data · sumber Yahoo Finance</p>

                    {/* Date pickers untuk pratinjau grafik */}
                    <div className="flex gap-2 mb-4">
                        <div className="flex-1">
                            <label className="text-xs text-gray-400 mb-1 block">Mulai</label>
                            <input
                                type="date"
                                value={startDate}
                                min={MIN_DATE}
                                max={endDate || TODAY}
                                onChange={(e) => setStartDate(e.target.value)}
                                className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:border-indigo-400"
                            />
                        </div>
                        <div className="flex-1">
                            <label className="text-xs text-gray-400 mb-1 block">Akhir</label>
                            <input
                                type="date"
                                value={endDate}
                                min={startDate || MIN_DATE}
                                max={TODAY}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:border-indigo-400"
                            />
                        </div>
                    </div>

                    {histLoading ? (
                        <div className="h-52 bg-gray-100 rounded-xl animate-pulse" />
                    ) : historical.length === 0 ? (
                        <div className="h-52 bg-gray-50 rounded-xl flex items-center justify-center text-sm text-gray-400">
                            {dateRangeInvalid ? "Pilih rentang tanggal yang valid." : "Gagal memuat data."}
                        </div>
                    ) : (
                        <ResponsiveContainer width="100%" height={208}>
                            <AreaChart data={historical} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                                <defs>
                                    <linearGradient id="silverGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis
                                    dataKey="date"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(v: string) => v.slice(5)}
                                    interval={tickInterval}
                                />
                                <YAxis
                                    tick={{ fontSize: 10 }}
                                    domain={["auto", "auto"]}
                                    tickFormatter={(v: number) => `$${v.toFixed(0)}`}
                                    width={48}
                                />
                                <Tooltip
                                    formatter={(v: any) => [`$${Number(v).toFixed(2)}`, "Harga Perak"]}
                                    labelFormatter={(l: any) => `Tanggal: ${l}`}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="price"
                                    stroke="#6366f1"
                                    fill="url(#silverGrad)"
                                    strokeWidth={2}
                                    dot={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    )}

                    <div className="grid grid-cols-2 gap-4 mt-4">
                        <div className="bg-gray-50 rounded-xl p-4">
                            <p className="text-xs text-gray-400 mb-1">Data Ditampilkan</p>
                            <p className="font-semibold text-gray-800">
                                {histLoading ? "..." : `${historical.length} hari`}
                            </p>
                        </div>
                        <div className="bg-gray-50 rounded-xl p-4">
                            <p className="text-xs text-gray-400 mb-1">Periode Prediksi</p>
                            <p className="font-semibold text-gray-800">{period} Hari</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
