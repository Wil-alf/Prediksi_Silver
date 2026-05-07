import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export const api = axios.create({ baseURL: BASE_URL });

export interface ModelMetrics {
    rmse: number;
    mae: number;
    mape: number;
    r2: number;
}

export interface TestComparisonRow {
    date: string;
    actual: number;
    xgboost: number;
    lightgbm: number;
}

export interface FutureForecastRow {
    date: string;
    xgboost: number;
    lightgbm: number;
}

export interface HistoricalRow {
    date: string;
    price: number;
}

export interface PredictResponse {
    period: 7 | 30;
    last_actual_date: string;
    xgboost: ModelMetrics;
    lightgbm: ModelMetrics;
    test_comparison: TestComparisonRow[];
    future_forecast: FutureForecastRow[];
    future_actual: HistoricalRow[];
    historical: HistoricalRow[];
    usd_to_idr: number;
}

export async function runPrediction(period: 7 | 30, endDate?: string): Promise<PredictResponse> {
    const body: Record<string, unknown> = { period };
    if (endDate) body.end_date = endDate;
    const { data } = await api.post<PredictResponse>("/api/predict", body);
    return data;
}

export async function getHistoricalData(startDate?: string, endDate?: string): Promise<HistoricalRow[]> {
    const params: Record<string, string> = {};
    if (startDate) params.start_date = startDate;
    if (endDate)   params.end_date   = endDate;
    const { data } = await api.get<HistoricalRow[]>("/api/historical", { params });
    return data;
}

// ── V2: Train-once, predict-many ─────────────────────────────────────

export interface ModelStatusV2 {
    trained: boolean;
    last_actual_date?: string;
    trained_at?: string;
}

export async function getModelStatusV2(): Promise<ModelStatusV2> {
    const { data } = await api.get<ModelStatusV2>("/api/v2/model-status");
    return data;
}

export async function trainModelV2(endDate?: string): Promise<{ status: string; trained_at: string }> {
    const body: Record<string, unknown> = {};
    if (endDate) body.end_date = endDate;
    const { data } = await api.post("/api/v2/train", body);
    return data;
}

export async function runPredictionV2(period: 7 | 30, endDate?: string): Promise<PredictResponse> {
    const body: Record<string, unknown> = { period };
    if (endDate) body.end_date = endDate;
    const { data } = await api.post<PredictResponse>("/api/v2/predict", body);
    return data;
}
