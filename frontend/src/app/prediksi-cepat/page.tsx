"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Play, Zap, RefreshCw, CheckCircle, AlertCircle } from "lucide-react";
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import axios from "axios";
import {
    getModelStatusV2,
    trainModelV2,
    runPredictionV2,
    getHistoricalData,
    type HistoricalRow,
    type ModelStatusV2,
} from "@/lib/api";

function formatDate(d: Date): string {
    return d.toISOString().slice(0, 10);
}

const TODAY    = formatDate(new Date());
const MIN_DATE = (() => { const d = new Date(); d.setFullYear(d.getFullYear() - 10); return formatDate(d); })();

function formatTrainedAt(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
}

export default function PrediksiCepatPage() {
    const [period, setPeriod]           = useState<7 | 30>(7);
    const [endDate, setEndDate]         = useState(TODAY);
    const [startDate, setStartDate]     = useState(MIN_DATE);
    const [historical, setHistorical]   = useState<HistoricalRow[]>([]);
    const [histLoading, setHistLoading] = useState(true);
    const [modelStatus, setModelStatus] = useState<ModelStatusV2>({ trained: false });
    const [statusLoading, setStatusLoading] = useState(true);
    const [training, setTraining]       = useState(false);
    const [predicting, setPredicting]   = useState(false);
    const router = useRouter();

    // Load model status on mount
    useEffect(() => {
        getModelStatusV2()
            .then(setModelStatus)
            .catch(console.error)
            .finally(() => setStatusLoading(false));
    }, []);

    // Historical chart with debounce
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

    const handleTrain = async () => {
        setTraining(true);
        try {
            await trainModelV2();
            const status = await getModelStatusV2();
            setModelStatus(status);
        } catch (err: unknown) {
            let msg = "Tidak diketahui.";
            if (axios.isAxiosError(err)) {
                if (err.code === "ECONNABORTED") msg = "Timeout — pelatihan melebihi batas waktu koneksi.";
                else if (err.response) msg = `Server error ${err.response.status}: ${JSON.stringify(err.response.data)}`;
                else msg = `Tidak dapat terhubung ke backend (${err.message}).`;
            } else if (err instanceof Error) {
                msg = err.message;
            }
            alert(`Gagal melatih model:\n${msg}`);
        } finally {
            setTraining(false);
        }
    };

    const handlePredict = async () => {
        setPredicting(true);
        try {
            const result = await runPredictionV2(period, endDate || undefined);
            sessionStorage.setItem("predictionResult", JSON.stringify(result));
            router.push("/hasil");
        } catch (err: unknown) {
            let msg = "Tidak diketahui.";
            if (axios.isAxiosError(err)) {
                if (err.code === "ECONNABORTED") msg = "Timeout — prediksi melebihi batas waktu koneksi.";
                else if (err.response) msg = `Server error ${err.response.status}: ${JSON.stringify(err.response.data)}`;
                else msg = `Tidak dapat terhubung ke backend (${err.message}).`;
            } else if (err instanceof Error) {
                msg = err.message;
            }
            alert(`Gagal menjalankan prediksi:\n${msg}`);
        } finally {
            setPredicting(false);
        }
    };

    const dateRangeInvalid = !startDate || !endDate || startDate >= endDate;
    const tickInterval     = historical.length > 0 ? Math.floor(historical.length / 6) : 30;

    return (
        <div className="max-w-6xl mx-auto px-6 py-10 bg-gray-50">
            <Link href="/" className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-900 mb-6">
                <ChevronLeft size={16} /> Kembali ke Beranda
            </Link>

            <div className="flex items-center gap-3 mb-1">
                <h1 className="text-3xl font-bold text-gray-800">Prediksi</h1>
            </div>
            <p className="text-gray-500 mb-8">
                Latih model sekali, prediksi berulang kali dalam hitungan detik.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left Panel */}
                <div className="flex flex-col gap-4">

                    {/* Status Card */}
                    <div className="bg-white rounded-2xl border border-gray-200 p-6">
                        <h2 className="font-semibold text-lg mb-4 text-gray-800">Status Model</h2>

                        {statusLoading ? (
                            <div className="h-10 bg-gray-100 rounded-lg animate-pulse" />
                        ) : modelStatus.trained ? (
                            <div className="flex items-start gap-3">
                                <CheckCircle size={20} className="text-emerald-500 shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-sm font-medium text-gray-800">Model siap digunakan</p>
                                    <p className="text-xs text-gray-500 mt-0.5">
                                        Data hingga: <span className="font-medium">{modelStatus.last_actual_date}</span>
                                    </p>
                                    <p className="text-xs text-gray-400">
                                        Dilatih: {modelStatus.trained_at ? formatTrainedAt(modelStatus.trained_at) : "—"}
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="flex items-start gap-3">
                                <AlertCircle size={20} className="text-amber-500 shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-sm font-medium text-gray-800">Model belum dilatih</p>
                                    <p className="text-xs text-gray-500 mt-0.5">Latih model terlebih dahulu sebelum prediksi.</p>
                                </div>
                            </div>
                        )}

                        <div className="mt-4 border-t border-gray-100 pt-4">
                            <button
                                onClick={handleTrain}
                                disabled={training}
                                className="w-full inline-flex items-center justify-center gap-2 border border-indigo-200 text-indigo-600 bg-indigo-50 py-2.5 rounded-lg text-sm font-medium hover:bg-indigo-100 transition disabled:opacity-60"
                            >
                                <RefreshCw size={15} className={training ? "animate-spin" : ""} />
                                {training ? "Melatih model... estimasi 3–5 menit" : (modelStatus.trained ? "Latih Ulang Model" : "Latih Model Sekarang")}
                            </button>
                            {training && (
                                <p className="text-xs text-gray-400 text-center mt-2">
                                    Melatih...
                                </p>
                            )}
                        </div>
                    </div>

                    {/* Prediction Parameters */}
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

                        <p className="text-sm text-gray-800 mb-1">Titik Awal Prediksi</p>
                        <p className="text-xs text-gray-400 mb-3">
                            Model akan memprediksi hari-hari setelah tanggal ini
                        </p>
                        <input
                            type="date"
                            value={endDate}
                            min={MIN_DATE}
                            max={TODAY}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:border-indigo-400 mb-6"
                        />

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

                        <button
                            onClick={handlePredict}
                            disabled={predicting || !modelStatus.trained || training}
                            className="w-full inline-flex items-center justify-center gap-2 bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 transition disabled:opacity-60"
                        >
                            <Play size={16} />
                            {predicting ? "Memproses..." : "Prediksi Cepat"}
                        </button>

                        {!modelStatus.trained && !statusLoading && (
                            <p className="text-xs text-amber-600 text-center mt-2">
                                Latih model terlebih dahulu untuk mengaktifkan prediksi.
                            </p>
                        )}
                    </div>
                </div>

                {/* Historical Chart */}
                <div className="bg-white rounded-2xl border border-gray-200 p-6">
                    <h2 className="font-semibold text-lg mb-1 text-gray-800">Data Historis Harga Perak</h2>
                    <p className="text-xs text-gray-500 mb-3">Sumber: Yahoo Finance</p>

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
                                    <linearGradient id="silverGradV2" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.3} />
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
                                    fill="url(#silverGradV2)"
                                    strokeWidth={2}
                                    dot={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    )}

                    <div className="grid grid-cols-2 gap-4 mt-4 mb-3">
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
                </div>
            </div>
        </div>
    );
}
