"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";
import type { PredictResponse } from "@/lib/api";

type Currency = "USD" | "IDR";

// Format helpers

function fmtPrice(usd: number, currency: Currency, rate: number): string {
    if (currency === "USD") return `$${usd.toFixed(2)}`;
    return `Rp ${Math.round(usd * rate).toLocaleString("id-ID")}`;
}

function fmtAxis(usd: number, currency: Currency, rate: number): string {
    if (currency === "USD") return `$${usd.toFixed(0)}`;
    const idr = usd * rate;
    return `Rp${(idr / 1000).toFixed(0)}rb`;
}

// RMSE, MAE, MAPE & R²
function fmtMetric(val: number, isPrice: boolean, currency: Currency, rate: number): string {
    if (!isPrice) return val.toFixed(4);
    if (currency === "USD") return `$${val.toFixed(2)}`;
    return `Rp ${Math.round(val * rate).toLocaleString("id-ID")}`;
}

// Components
function MetricCard({
    label, value, unit = "", isPrice = false, currency, rate,
}: {
    label: string; value: number | null; unit?: string;
    isPrice?: boolean; currency: Currency; rate: number;
}) {
    return (
        <div className="bg-gray-50 rounded-xl p-4">
            <p className="text-xs text-gray-400 mb-1">{label}</p>
            <p className="text-xl font-semibold text-gray-800">
                {value !== null ? `${fmtMetric(value, isPrice, currency, rate)}${unit}` : "–"}
            </p>
        </div>
    );
}

function CurrencyToggle({ value, onChange }: { value: Currency; onChange: (c: Currency) => void }) {
    return (
        <div className="flex rounded-lg border border-gray-200 overflow-hidden text-sm font-medium">
            {(["USD", "IDR"] as Currency[]).map((c) => (
                <button
                    key={c}
                    onClick={() => onChange(c)}
                    className={`px-4 py-1.5 transition ${
                        value === c
                            ? "bg-indigo-600 text-white"
                            : "text-gray-500 hover:bg-gray-50"
                    }`}
                >
                    {c}
                </button>
            ))}
        </div>
    );
}

const MODEL_LABELS: Record<string, string> = {
    actual:   "Aktual",
    xgboost:  "Prophet+XGBoost",
    lightgbm: "Prophet+LightGBM",
};

// Page 

export default function HasilPage() {
    const [result, setResult]     = useState<PredictResponse | null>(null);
    const [currency, setCurrency] = useState<Currency>("IDR");

    useEffect(() => {
        const stored = sessionStorage.getItem("predictionResult");
        if (stored) setResult(JSON.parse(stored));
    }, []);

    const rate = result?.usd_to_idr ?? 15800;
    const xgb  = result?.xgboost;
    const lgbm = result?.lightgbm;

    const hasActualForecast = (result?.future_actual?.length ?? 0) > 0;

    const futureActualMap: Record<string, number> = Object.fromEntries(
        (result?.future_actual ?? []).map((r) => [r.date, r.price])
    );
    const futureChartData = (result?.future_forecast ?? []).map((row) => ({
        ...row,
        actual: futureActualMap[row.date] ?? null,
    }));

    const futureTickInterval = result?.future_forecast?.length
        ? Math.max(0, Math.floor(result.future_forecast.length / 6) - 1)
        : 0;
    const testTickInterval = result?.test_comparison?.length
        ? Math.floor(result.test_comparison.length / 5)
        : 5;

    const tooltipFmt = (v: any, name: any) => [
        v !== null ? fmtPrice(Number(v), currency, rate) : "–",
        MODEL_LABELS[name] ?? name,
    ];

    return (
        <div className="max-w-6xl mx-auto px-6 py-10">
            <Link href="/prediksi-cepat" className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-900 mb-6">
                <ChevronLeft size={16} /> Kembali ke Prediksi
            </Link>

            {/* Header + toggle */}
            <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
                <div>
                    <h1 className="text-3xl font-bold mb-1">Hasil Prediksi</h1>
                    <p className="text-gray-500">
                        {result
                            ? `Prediksi ${result.period} hari ke depan · berdasarkan data hingga ${result.last_actual_date}`
                            : "Menunggu data prediksi..."}
                    </p>
                    {result && (
                        <p className="text-xs text-gray-400 mt-1">
                            Kurs: 1 USD = Rp {Math.round(rate).toLocaleString("id-ID")}
                        </p>
                    )}
                </div>
                {result && <CurrencyToggle value={currency} onChange={setCurrency} />}
            </div>

            {!result && (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-8 text-sm text-amber-700">
                    Belum ada data prediksi. Silakan jalankan prediksi terlebih dahulu di{" "}
                    <Link href="/prediksi-cepat" className="underline font-medium">halaman prediksi</Link>.
                </div>
            )}

            {/* PREDIKSI MASA DEPAN grafik */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6 mb-6">
                <div className="flex items-center justify-between mb-1">
                    <h2 className="font-semibold text-lg">Prediksi Harga Perak ke Depan</h2>
                    {result && (
                        <span className="text-xs bg-indigo-50 text-indigo-600 font-medium px-3 py-1 rounded-full">
                            {result.period} hari ke depan
                        </span>
                    )}
                </div>
                <p className="text-xs text-gray-400 mb-4">
                    {result?.future_forecast?.length
                        ? `${result.future_forecast[0].date} sampai dengan ${result.future_forecast[result.future_forecast.length - 1].date}`
                        : "Rentang tanggal prediksi"}
                </p>

                {futureChartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={futureChartData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                            <XAxis
                                dataKey="date"
                                tick={{ fontSize: 10 }}
                                tickFormatter={(v: string) => v.slice(5)}
                                interval={futureTickInterval}
                            />
                            <YAxis
                                tick={{ fontSize: 10 }}
                                domain={["auto", "auto"]}
                                tickFormatter={(v: number) => fmtAxis(v, currency, rate)}
                                width={currency === "IDR" ? 72 : 52}
                            />
                            <Tooltip formatter={tooltipFmt} labelFormatter={(l: any) => `Tanggal: ${l}`} />
                            <Legend formatter={(v: string) => MODEL_LABELS[v] ?? v} />
                            {hasActualForecast && (
                                <Line type="monotone" dataKey="actual" stroke="#64748b" strokeWidth={2}
                                    dot={{ r: 4 }} activeDot={{ r: 6 }} connectNulls={false} />
                            )}
                            <Line type="monotone" dataKey="xgboost"  stroke="#6366f1" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                            <Line type="monotone" dataKey="lightgbm" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="h-52 bg-gray-50 rounded-xl flex items-center justify-center text-sm text-gray-400">
                        Grafik akan ditampilkan setelah prediksi selesai
                    </div>
                )}
            </div>

            {/* TABEL PREDIKSI MASA DEPAN */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6 mb-8">
                <h2 className="font-semibold text-lg mb-1">Tabel Prediksi</h2>
                <p className="text-xs text-gray-400 mb-4">
                    Estimasi harga perak per troy ounce dalam {currency === "IDR" ? "Rupiah (IDR)" : "Dolar AS (USD)"}
                </p>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-gray-100">
                                {[
                                    "Tanggal",
                                    ...(hasActualForecast ? ["Harga Aktual"] : []),
                                    "Prophet+XGBoost",
                                    "Prophet+LightGBM",
                                    ...(hasActualForecast ? ["Selisih XGB"] : []),
                                    ...(hasActualForecast ? ["Selisih LGBM"] : []),
                                ].map((h) => (
                                    <th key={h} className="text-left py-2 pr-6 text-gray-500 font-medium">{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {futureChartData.length > 0 ? (
                                futureChartData.map((row, i) => {
                                    const diff1 = row.actual !== null ? Math.abs((row.actual as number) - row.xgboost) : null;
                                    const diff2 = row.actual !== null ? Math.abs((row.actual as number) - row.lightgbm) : null;
                                    return (
                                        <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                                            <td className="py-2 pr-6 font-medium">{row.date}</td>
                                            {hasActualForecast && (
                                                <td className="py-2 pr-6 text-gray-700 font-medium">
                                                    {row.actual !== null ? fmtPrice(row.actual as number, currency, rate) : "–"}
                                                </td>
                                            )}
                                            <td className="py-2 pr-6 text-indigo-700 font-medium">
                                                {fmtPrice(row.xgboost, currency, rate)}
                                            </td>
                                            <td className="py-2 pr-6 text-emerald-700 font-medium">
                                                {fmtPrice(row.lightgbm, currency, rate)}
                                            </td>
                                            {hasActualForecast && (
                                                <td className="py-2 pr-6 text-gray-400">
                                                    {diff1 !== null ? fmtPrice(diff1, currency, rate) : "–"}
                                                </td>
                                            )}
                                            {hasActualForecast && (
                                                <td className="py-2 pr-6 text-gray-400">
                                                    {diff2 !== null ? fmtPrice(diff2, currency, rate) : "–"}
                                                </td>
                                            )}
                                        </tr>
                                    );
                                })
                            ) : (
                                <tr>
                                    <td colSpan={hasActualForecast ? 6 : 3} className="py-6 text-center text-gray-400 text-sm">
                                        Tabel akan ditampilkan setelah prediksi dijalankan
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* METRIK EVALUASI */}
            <h2 className="font-semibold text-xl mb-1">Metrik Evaluasi Model</h2>
            <p className="text-xs text-gray-400 mb-4">
                Dihitung pada 30 hari terakhir data historis (set pengujian)
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {[
                    { label: "Prophet + XGBoost",  tag: "Model 1", color: "indigo",  data: xgb  },
                    { label: "Prophet + LightGBM", tag: "Model 2", color: "green", data: lgbm },
                ].map(({ label, tag, color, data }) => (
                    <div key={label} className="bg-white rounded-2xl border border-gray-200 p-6">
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`w-2.5 h-2.5 rounded-full bg-${color}-500`} />
                            <h3 className="font-semibold text-lg">{label}</h3>
                        </div>
                        <p className="text-xs text-gray-400 mb-5">{tag}</p>
                        <div className="grid gap-3">
                            <MetricCard label="MAE (Mean Absolute Error)"              value={data?.mae  ?? null} isPrice currency={currency} rate={rate} />
                            <MetricCard label="RMSE (Root Mean Square Error)"          value={data?.rmse ?? null} isPrice currency={currency} rate={rate} />
                            <MetricCard label="MAPE (Mean Absolute Percentage Error)"  value={data?.mape ?? null} unit="%" currency={currency} rate={rate} />
                            <MetricCard label="R² Score"                               value={data?.r2   ?? null} currency={currency} rate={rate} />
                        </div>
                    </div>
                ))}
            </div>

            {/* VALIDASI HISTORIS grafik */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6 mb-6">
                <h2 className="font-semibold text-lg mb-1">Validasi Model (Perbandingan Historis)</h2>
                <p className="text-xs text-gray-400 mb-4">
                    Prediksi vs harga aktual pada 30 hari terakhir data (set pengujian)
                </p>
                {result?.test_comparison?.length ? (
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={result.test_comparison} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                            <XAxis
                                dataKey="date"
                                tick={{ fontSize: 10 }}
                                tickFormatter={(v: string) => v.slice(5)}
                                interval={testTickInterval}
                            />
                            <YAxis
                                tick={{ fontSize: 10 }}
                                domain={["auto", "auto"]}
                                tickFormatter={(v: number) => fmtAxis(v, currency, rate)}
                                width={currency === "IDR" ? 72 : 52}
                            />
                            <Tooltip formatter={tooltipFmt} labelFormatter={(l: any) => `Tanggal: ${l}`} />
                            <Legend formatter={(v: string) => MODEL_LABELS[v] ?? v} />
                            <Line type="monotone" dataKey="actual"   stroke="#64748b" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="xgboost"  stroke="#6366f1" strokeWidth={2} dot={false} strokeDasharray="5 3" />
                            <Line type="monotone" dataKey="lightgbm" stroke="#10b981" strokeWidth={2} dot={false} strokeDasharray="2 2" />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="h-52 bg-gray-50 rounded-xl flex items-center justify-center text-sm text-gray-400">
                        Grafik validasi akan ditampilkan setelah prediksi selesai
                    </div>
                )}
            </div>

            {/* TABEL VALIDASI */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
                <h2 className="font-semibold text-lg mb-1">Tabel Validasi Historis</h2>
                <p className="text-xs text-gray-400 mb-4">
                    30 hari set pengujian dalam {currency === "IDR" ? "Rupiah (IDR)" : "Dolar AS (USD)"}
                </p>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-gray-100">
                                {["Tanggal", "Harga Aktual", "Prophet+XGBoost", "Prophet+LightGBM"].map((h) => (
                                    <th key={h} className="text-left py-2 pr-6 text-gray-500 font-medium">{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {result?.test_comparison?.length ? (
                                result.test_comparison.map((row, i) => (
                                    <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                                        <td className="py-2 pr-6">{row.date}</td>
                                        <td className="py-2 pr-6 font-medium">
                                            {fmtPrice(row.actual, currency, rate)}
                                        </td>
                                        <td className="py-2 pr-6 text-indigo-600">
                                            {fmtPrice(row.xgboost, currency, rate)}
                                        </td>
                                        <td className="py-2 pr-6 text-emerald-600">
                                            {fmtPrice(row.lightgbm, currency, rate)}
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={4} className="py-6 text-center text-gray-400 text-sm">
                                        Tabel validasi akan ditampilkan setelah prediksi dijalankan
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="mt-8 flex justify-center">
                <Link
                    href="/prediksi-cepat"
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors"
                >
                    <ChevronLeft size={16} /> Kembali ke Prediksi
                </Link>
            </div>
        </div>
    );
}
