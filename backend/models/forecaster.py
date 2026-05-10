import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
from datetime import datetime as _dt

import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_dict, model_from_dict
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

TICKERS = {"silver": "SI=F", "gold": "GC=F", "oil": "CL=F", "usd": "DX-Y.NYB"}
LAG_DAYS        = [1, 2, 3, 5, 7, 10, 14]
ROLLING_WINDOWS = [5, 10, 20, 30]
TEST_SIZE       = 30   # jumlah baris yang ditampilkan di chart
SERIES_LIST     = ["silver", "gold", "oil", "usd"]
HIST_TAIL       = 100  # baris terakhir yang disimpan untuk konstruksi fitur saat inference
SAVE_DIR        = os.path.join(os.path.dirname(__file__), "saved")


def fetch_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    ticker_list = list(TICKERS.values())
    kwargs: dict = {"interval": "1d", "auto_adjust": True, "progress": False}
    if end_date:
        end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        kwargs["start"] = start_date or "1990-01-01"
        kwargs["end"]   = end_dt
    elif start_date:
        kwargs["start"] = start_date
    else:
        kwargs["period"] = "10y"
    raw  = yf.download(ticker_list, **kwargs)
    close = raw["Close"].copy()
    rename_map = {v: k for k, v in TICKERS.items()}
    close = close.rename(columns=rename_map)[SERIES_LIST]
    close = close.dropna().reset_index()
    if "Date" in close.columns:
        close = close.rename(columns={"Date": "ds"})
    elif "Datetime" in close.columns:
        close = close.rename(columns={"Datetime": "ds"})
    else:
        close.columns = ["ds"] + list(close.columns[1:])
    close["ds"] = pd.to_datetime(close["ds"])
    if close["ds"].dt.tz is not None:
        close["ds"] = close["ds"].dt.tz_localize(None)
    return close.reset_index(drop=True)


def _ema_arr(arr: np.ndarray, span: int) -> float:
    """EMA over a numpy array"""
    alpha = 2.0 / (span + 1)
    ema   = float(arr[0])
    for v in arr[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return ema


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df       = df.copy()
    new_cols: dict = {}

    for series in SERIES_LIST:
        ret = df[series].pct_change()
        for lag in LAG_DAYS:
            new_cols[f"{series}_lag_{lag}"]     = df[series].shift(lag)
            new_cols[f"{series}_ret_lag_{lag}"] = ret.shift(lag)
        for w in ROLLING_WINDOWS:
            new_cols[f"{series}_roll_mean_{w}"] = df[series].rolling(w).mean()
            new_cols[f"{series}_roll_std_{w}"]  = df[series].rolling(w).std()

    # Technical indicators for silver
    delta    = df["silver"].diff()
    up       = delta.clip(lower=0)
    down     = (-delta).clip(lower=0)
    avg_up   = up.rolling(14).mean()
    avg_down = down.rolling(14).mean().replace(0, 1e-9)
    new_cols["silver_rsi_14"] = (100 - 100 / (1 + avg_up / avg_down)).shift(1)

    ema12 = df["silver"].ewm(span=12, adjust=False).mean()
    ema26 = df["silver"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    new_cols["silver_macd"]        = macd.shift(1)
    new_cols["silver_macd_signal"] = macd.ewm(span=9, adjust=False).mean().shift(1)

    roll_mean_20 = df["silver"].rolling(20).mean()
    roll_std_20  = df["silver"].rolling(20).std().replace(0, 1e-9)
    new_cols["silver_bb_pos"] = ((df["silver"] - roll_mean_20) / roll_std_20).shift(1)
    new_cols["silver_atr_14"] = df["silver"].diff().abs().rolling(14).mean().shift(1)

    # Cross-asset ratio features (4 variables, lagged)
    gs_ratio = df["gold"]   / df["silver"].replace(0, 1e-9)
    so_ratio = df["silver"] / df["oil"].replace(0, 1e-9)
    new_cols["gold_silver_ratio_lag_1"] = gs_ratio.shift(1)
    new_cols["gold_silver_ratio_lag_5"] = gs_ratio.shift(5)
    new_cols["silver_oil_ratio_lag_1"]  = so_ratio.shift(1)

    new_cols["day_of_week"] = pd.to_datetime(df["ds"]).dt.dayofweek
    new_cols["month"]       = pd.to_datetime(df["ds"]).dt.month

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.dropna().reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae  = float(mean_absolute_error(actual, predicted))
    safe = np.where(actual == 0, 1e-9, actual)
    mape = float(np.mean(np.abs((actual - predicted) / safe)) * 100)
    r2   = float(r2_score(actual, predicted))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _fit_prophet(df_train: pd.DataFrame) -> Prophet:
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.15,
        seasonality_mode="multiplicative",
        uncertainty_samples=0,
    )
    m.fit(df_train[["ds", "silver"]].rename(columns={"silver": "y"}))
    return m


def _get_prophet_components(m: Prophet, dates: pd.DataFrame) -> pd.DataFrame:
    fc  = m.predict(dates)
    out = fc[["ds", "yhat", "trend"]].copy()
    out["weekly"] = fc["weekly"] if "weekly" in fc.columns else 0.0
    out["yearly"] = fc["yearly"] if "yearly" in fc.columns else 0.0
    return out.reset_index(drop=True)


def _feature_row(
    silver_hist: list,
    gold_hist: list,
    oil_hist: list,
    usd_hist: list,
    ph_yhat: float,
    ph_trend: float,
    ph_weekly: float,
    ph_yearly: float,
    fdate: pd.Timestamp,
    feature_cols: list,
) -> np.ndarray:
    row = {}
    silver_arr = np.array(silver_hist, dtype=float)
    gold_arr   = np.array(gold_hist,   dtype=float)
    oil_arr    = np.array(oil_hist,    dtype=float)
    usd_arr    = np.array(usd_hist,    dtype=float)

    for series, arr in [
        ("silver", silver_arr), ("gold", gold_arr),
        ("oil",    oil_arr),    ("usd",  usd_arr),
    ]:
        for lag in LAG_DAYS:
            row[f"{series}_lag_{lag}"] = arr[-lag] if len(arr) >= lag else arr[0]
        for lag in LAG_DAYS:
            if len(arr) > lag:
                p1, p0 = arr[-lag], arr[-(lag + 1)]
                row[f"{series}_ret_lag_{lag}"] = (p1 - p0) / p0 if p0 != 0 else 0.0
            else:
                row[f"{series}_ret_lag_{lag}"] = 0.0
        for w in ROLLING_WINDOWS:
            tail = arr[-w:]
            row[f"{series}_roll_mean_{w}"] = float(np.mean(tail))
            row[f"{series}_roll_std_{w}"]  = float(np.std(tail)) if len(tail) > 1 else 0.0

    # RSI(14)
    if len(silver_arr) >= 15:
        deltas   = np.diff(silver_arr[-15:])
        avg_up   = float(np.mean(np.where(deltas > 0, deltas, 0.0)))
        avg_down = float(np.mean(np.where(deltas < 0, -deltas, 0.0)))
        avg_down = avg_down if avg_down > 0 else 1e-9
        row["silver_rsi_14"] = 100.0 - 100.0 / (1.0 + avg_up / avg_down)
    else:
        row["silver_rsi_14"] = 50.0

    # MACD line
    if len(silver_arr) >= 26:
        n = min(len(silver_arr), 100)
        s = silver_arr[-n:]
        row["silver_macd"] = _ema_arr(s, 12) - _ema_arr(s, 26)
        if len(silver_arr) >= 35:
            macd_hist = []
            for back in range(19, -1, -1):
                s_sub = silver_arr[:len(silver_arr) - back] if back > 0 else silver_arr
                if len(s_sub) >= 26:
                    sl = s_sub[-min(len(s_sub), 100):]
                    macd_hist.append(_ema_arr(sl, 12) - _ema_arr(sl, 26))
            row["silver_macd_signal"] = _ema_arr(np.array(macd_hist), 9) if macd_hist else row["silver_macd"]
        else:
            row["silver_macd_signal"] = row["silver_macd"]
    else:
        row["silver_macd"]        = 0.0
        row["silver_macd_signal"] = 0.0

    # Bollinger Band position
    if len(silver_arr) >= 20:
        m20 = float(np.mean(silver_arr[-20:]))
        s20 = float(np.std(silver_arr[-20:]))
        row["silver_bb_pos"] = (silver_arr[-1] - m20) / (s20 if s20 > 0 else 1e-9)
    else:
        row["silver_bb_pos"] = 0.0

    # ATR(14)
    if len(silver_arr) >= 15:
        row["silver_atr_14"] = float(np.mean(np.abs(np.diff(silver_arr[-15:]))))
    else:
        row["silver_atr_14"] = 0.0

    # Cross-asset ratios
    sv1 = float(silver_arr[-1]) if silver_arr[-1] != 0 else 1e-9
    oi1 = float(oil_arr[-1])    if len(oil_arr) >= 1 and oil_arr[-1] != 0 else 1e-9

    row["gold_silver_ratio_lag_1"] = float(gold_arr[-1]) / sv1 if len(gold_arr) >= 1 else 1.0
    if len(gold_arr) >= 5 and len(silver_arr) >= 5:
        sv5 = float(silver_arr[-5]) if silver_arr[-5] != 0 else 1e-9
        row["gold_silver_ratio_lag_5"] = float(gold_arr[-5]) / sv5
    else:
        row["gold_silver_ratio_lag_5"] = row["gold_silver_ratio_lag_1"]
    row["silver_oil_ratio_lag_1"] = sv1 / oi1

    row["day_of_week"] = fdate.dayofweek
    row["month"]       = fdate.month
    row["ph_yhat"]     = ph_yhat
    row["ph_trend"]    = ph_trend
    row["ph_weekly"]   = ph_weekly
    row["ph_yearly"]   = ph_yearly
    return np.array([row[col] for col in feature_cols])


def _build_boosting_models(X: np.ndarray, y: np.ndarray):
    xgb = XGBRegressor(
        n_estimators=1200, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        random_state=42, verbosity=0,
    )
    xgb.fit(X, y)
    lgbm = LGBMRegressor(
        n_estimators=2000, learning_rate=0.005, max_depth=6, num_leaves=63,
        subsample=0.8, subsample_freq=5, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=2.0, min_child_samples=15,
        random_state=42, verbose=-1,
    )
    lgbm.fit(X, y)
    return xgb, lgbm


def run_forecast(period: int = 7, start_date: str = None, end_date: str = None) -> dict:
    raw = fetch_data(start_date, end_date)
    df  = build_features(raw)

    test_size = max(TEST_SIZE, int(len(df) * 0.2))

    train = df.iloc[:-test_size].copy().reset_index(drop=True)
    test  = df.iloc[-test_size:].copy().reset_index(drop=True)

    price_lag_cols = [f"{s}_lag_{d}"     for s in SERIES_LIST for d in LAG_DAYS]
    ret_lag_cols   = [f"{s}_ret_lag_{d}" for s in SERIES_LIST for d in LAG_DAYS]
    roll_cols      = [
        f"{s}_{fn}_{w}"
        for s in SERIES_LIST
        for fn in ["roll_mean", "roll_std"]
        for w in ROLLING_WINDOWS
    ]
    calendar_cols = ["day_of_week", "month"]
    prophet_cols  = ["ph_yhat", "ph_trend", "ph_weekly", "ph_yearly"]
    tech_cols     = ["silver_rsi_14", "silver_macd", "silver_macd_signal", "silver_bb_pos", "silver_atr_14"]
    ratio_cols    = ["gold_silver_ratio_lag_1", "gold_silver_ratio_lag_5", "silver_oil_ratio_lag_1"]
    feature_cols  = price_lag_cols + ret_lag_cols + roll_cols + calendar_cols + prophet_cols + tech_cols + ratio_cols

    # EVALUATION MODEL
    m_eval  = _fit_prophet(train)
    ph_eval = _get_prophet_components(m_eval, df[["ds"]].copy())
    for col in ["yhat", "trend", "weekly", "yearly"]:
        train[f"ph_{col}"] = ph_eval[col].values[:len(train)]
        test[f"ph_{col}"]  = ph_eval[col].values[len(train):]

    X_train = train[feature_cols].values
    log_ret_train = np.where(
        train["silver_lag_1"].values > 0,
        np.log(train["silver"].values / train["silver_lag_1"].values),
        0.0,
    )
    xgb_eval, lgbm_eval = _build_boosting_models(X_train, log_ret_train)

    # Rolling origin evaluation on test set
    n_total     = len(df)
    silver_hist = list(df["silver"].values[:-test_size])
    gold_hist   = list(df["gold"].values)
    oil_hist    = list(df["oil"].values)
    usd_hist    = list(df["usd"].values)
    actual_test = test["silver"].values

    xgb_test_preds  = []
    lgbm_test_preds = []
    for i in range(test_size):
        ph_row = test.iloc[i]
        fdate  = pd.to_datetime(ph_row["ds"])
        cutoff = n_total - test_size + i
        x = _feature_row(
            silver_hist,
            gold_hist[:cutoff], oil_hist[:cutoff], usd_hist[:cutoff],
            float(ph_row["ph_yhat"]), float(ph_row["ph_trend"]),
            float(ph_row["ph_weekly"]), float(ph_row["ph_yearly"]),
            fdate, feature_cols,
        )
        prev_price = silver_hist[-1]
        xgb_test_preds.append(prev_price * np.exp(float(xgb_eval.predict(x.reshape(1, -1))[0])))
        lgbm_test_preds.append(prev_price * np.exp(float(lgbm_eval.predict(x.reshape(1, -1))[0])))
        silver_hist.append(float(actual_test[i]))

    xgb_final  = np.array(xgb_test_preds)
    lgbm_final = np.array(lgbm_test_preds)

    # PRODUCTION MODEL
    m_prod  = _fit_prophet(df)
    ph_prod = _get_prophet_components(m_prod, df[["ds"]].copy())
    df_prod = df.copy()
    for col in ["yhat", "trend", "weekly", "yearly"]:
        df_prod[f"ph_{col}"] = ph_prod[col].values

    X_all = df_prod[feature_cols].values
    log_ret_all = np.where(
        df_prod["silver_lag_1"].values > 0,
        np.log(df_prod["silver"].values / df_prod["silver_lag_1"].values),
        0.0,
    )
    xgb_prod, lgbm_prod = _build_boosting_models(X_all, log_ret_all)

    # Generate future dates
    last_date     = df["ds"].max()
    future_range  = m_prod.make_future_dataframe(periods=period, freq="B")
    ph_future_all = _get_prophet_components(m_prod, future_range)
    ph_future = (
        ph_future_all[ph_future_all["ds"] > last_date]
        .head(period)
        .reset_index(drop=True)
    )

    # Iterative future prediction
    future_silver_hist_xgb  = list(df["silver"].values)
    future_silver_hist_lgbm = list(df["silver"].values)
    xgb_future_vals         = []
    lgbm_future_vals        = []

    for i in range(len(ph_future)):
        ph_row = ph_future.iloc[i]
        fdate  = pd.to_datetime(ph_row["ds"])

        x_xgb = _feature_row(
            future_silver_hist_xgb,
            gold_hist, oil_hist, usd_hist,
            float(ph_row["yhat"]),   float(ph_row["trend"]),
            float(ph_row["weekly"]), float(ph_row["yearly"]),
            fdate, feature_cols,
        )
        x_lgbm = _feature_row(
            future_silver_hist_lgbm,
            gold_hist, oil_hist, usd_hist,
            float(ph_row["yhat"]),   float(ph_row["trend"]),
            float(ph_row["weekly"]), float(ph_row["yearly"]),
            fdate, feature_cols,
        )

        xgb_price  = future_silver_hist_xgb[-1]  * np.exp(float(xgb_prod.predict(x_xgb.reshape(1, -1))[0]))
        lgbm_price = future_silver_hist_lgbm[-1] * np.exp(float(lgbm_prod.predict(x_lgbm.reshape(1, -1))[0]))

        xgb_future_vals.append(xgb_price)
        lgbm_future_vals.append(lgbm_price)
        future_silver_hist_xgb.append(xgb_price)
        future_silver_hist_lgbm.append(lgbm_price)

    # Assemble response
    test_comparison = [
        {
            "date":     str(test["ds"].iloc[i].date()),
            "actual":   float(actual_test[i]),
            "prophet":  float(test["ph_yhat"].iloc[i]),
            "xgboost":  float(xgb_final[i]),
            "lightgbm": float(lgbm_final[i]),
        }
        for i in range(TEST_SIZE)
    ]

    future_forecast = [
        {
            "date":     str(pd.to_datetime(ph_future.loc[i, "ds"]).date()),
            "prophet":  float(ph_future.loc[i, "yhat"]),
            "xgboost":  float(xgb_future_vals[i]),
            "lightgbm": float(lgbm_future_vals[i]),
        }
        for i in range(len(ph_future))
    ]
    future_actual = _fetch_future_actual(future_forecast, last_date)

    historical = [
        {"date": str(row["ds"].date()), "price": float(row["silver"])}
        for _, row in df.iterrows()
    ]

    usd_to_idr = _fetch_usd_to_idr()

    return {
        "period":           period,
        "last_actual_date": str(last_date.date()),
        "prophet":          _metrics(actual_test, test["ph_yhat"].values),
        "xgboost":          _metrics(actual_test, xgb_final),
        "lightgbm":         _metrics(actual_test, lgbm_final),
        "test_comparison":  test_comparison,
        "future_forecast":  future_forecast,
        "future_actual":    future_actual,
        "historical":       historical,
        "usd_to_idr":       usd_to_idr,
    }

def _fetch_usd_to_idr() -> float:
    """Ambil kurs USD/IDR terkini dari yfinance (ticker USDIDR=X)."""
    try:
        raw = yf.download("USDIDR=X", period="5d", interval="1d",
                          auto_adjust=True, progress=False)
        if not raw.empty:
            price = float(raw["Close"].dropna().iloc[-1])
            if price > 1000:
                return price
    except Exception:
        pass
    return 15800.0


def _fetch_future_actual(future_forecast: list, last_date: pd.Timestamp) -> list:
    if not future_forecast:
        return []
    today            = pd.Timestamp.now().normalize()
    last_forecast_dt = pd.to_datetime(future_forecast[-1]["date"])
    if last_forecast_dt.date() >= today.date():
        return []
    try:
        fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_end   = (last_forecast_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        act_raw = yf.download(
            "SI=F", start=fetch_start, end=fetch_end,
            interval="1d", auto_adjust=True, progress=False,
        )
        if act_raw.empty:
            return []
        if hasattr(act_raw.columns, "levels"):
            act_close = act_raw["Close"]
        else:
            act_close = act_raw["Close"] if "Close" in act_raw.columns else act_raw.iloc[:, 0]
        act_close = act_close.dropna().reset_index()
        act_close.columns = ["date", "price"]
        act_close["date"] = act_close["date"].astype(str).str[:10]
        forecast_dates = {r["date"] for r in future_forecast}
        return [
            {"date": str(r["date"]), "price": float(r["price"])}
            for _, r in act_close.iterrows()
            if str(r["date"]) in forecast_dates
        ]
    except Exception:
        return []



# TRAIN & SAVE


def _build_feature_cols() -> list:
    price_lag_cols = [f"{s}_lag_{d}"     for s in SERIES_LIST for d in LAG_DAYS]
    ret_lag_cols   = [f"{s}_ret_lag_{d}" for s in SERIES_LIST for d in LAG_DAYS]
    roll_cols      = [
        f"{s}_{fn}_{w}"
        for s in SERIES_LIST
        for fn in ["roll_mean", "roll_std"]
        for w in ROLLING_WINDOWS
    ]
    calendar_cols = ["day_of_week", "month"]
    prophet_cols  = ["ph_yhat", "ph_trend", "ph_weekly", "ph_yearly"]
    tech_cols     = ["silver_rsi_14", "silver_macd", "silver_macd_signal", "silver_bb_pos", "silver_atr_14"]
    ratio_cols    = ["gold_silver_ratio_lag_1", "gold_silver_ratio_lag_5", "silver_oil_ratio_lag_1"]
    return price_lag_cols + ret_lag_cols + roll_cols + calendar_cols + prophet_cols + tech_cols + ratio_cols


def is_model_saved() -> bool:
    required = ["xgb_prod.joblib", "lgbm_prod.joblib", "prophet_prod.json", "metadata.json"]
    return all(os.path.exists(os.path.join(SAVE_DIR, f)) for f in required)


def get_model_status() -> dict:
    if not is_model_saved():
        return {"trained": False}
    with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return {
        "trained":          True,
        "last_actual_date": meta["last_actual_date"],
        "trained_at":       meta["trained_at"],
        "start_date":       meta.get("start_date"),
        "end_date":         meta.get("end_date"),
    }


def train_and_save(start_date: str = None, end_date: str = None) -> dict:
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw = fetch_data(start_date, end_date)
    df  = build_features(raw)

    m_prod = _fit_prophet(df)
    ph_all = _get_prophet_components(m_prod, df[["ds"]].copy())
    for col in ["yhat", "trend", "weekly", "yearly"]:
        df[f"ph_{col}"] = ph_all[col].values

    feature_cols = _build_feature_cols()

    MAX_H  = 30
    prices = df["silver"].values
    n      = len(prices)
    Y      = np.full((n, MAX_H), np.nan)
    for h in range(1, MAX_H + 1):
        base = np.where(prices[:n - h] > 0, prices[:n - h], 1e-9)
        Y[:n - h, h - 1] = np.log(prices[h:] / base)

    feat_ok   = df[feature_cols].notna().all(axis=1).values
    targ_ok   = ~np.isnan(Y).any(axis=1)
    valid_mask = feat_ok & targ_ok
    df_v   = df[valid_mask].reset_index(drop=True)
    Y_v    = Y[valid_mask]
    X_all  = df_v[feature_cols].values
    prc_v  = df_v["silver"].values 

    n_v       = len(df_v)
    test_size = max(TEST_SIZE, int(n_v * 0.2))
    split     = n_v - test_size

    X_tr, X_te = X_all[:split], X_all[split:]
    Y_tr        = Y_v[:split]

    xgb_base = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        n_jobs=1, random_state=42, verbosity=0,
    )
    lgbm_base = LGBMRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=5, num_leaves=31,
        subsample=0.8, subsample_freq=5, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=2.0, min_child_samples=10,
        n_jobs=1, random_state=42, verbose=-1,
    )

    #  Eval models 
    print("  [1/4] Melatih XGBoost eval...")
    xgb_eval  = MultiOutputRegressor(xgb_base,  n_jobs=-1).fit(X_tr, Y_tr)
    print("  [2/4] Melatih LightGBM eval...")
    lgbm_eval = MultiOutputRegressor(lgbm_base, n_jobs=-1).fit(X_tr, Y_tr)

    xgb_preds, lgbm_preds, actuals, eval_dates = [], [], [], []
    for i in range(test_size):
        t = split + i
        if t == 0:
            continue
        prev_X = X_all[t - 1].reshape(1, -1)
        prev_p = float(prc_v[t - 1])
        act_p  = float(prc_v[t])

        xgb_preds.append(prev_p  * np.exp(float(xgb_eval.predict(prev_X)[0, 0])))
        lgbm_preds.append(prev_p * np.exp(float(lgbm_eval.predict(prev_X)[0, 0])))
        actuals.append(act_p)
        eval_dates.append(str(df_v.iloc[t]["ds"].date()))

    n_eval       = min(TEST_SIZE, len(actuals))
    actual_arr   = np.array(actuals[:n_eval])
    xgb_arr      = np.array(xgb_preds[:n_eval])
    lgbm_arr     = np.array(lgbm_preds[:n_eval])
    prophet_arr  = df_v.iloc[split:split + n_eval]["ph_yhat"].values

    xgboost_met  = _metrics(actual_arr, xgb_arr)
    lightgbm_met = _metrics(actual_arr, lgbm_arr)
    prophet_met  = _metrics(actual_arr, prophet_arr)

    test_comparison = [
        {
            "date":     eval_dates[i],
            "actual":   float(actuals[i]),
            "prophet":  float(prophet_arr[i]) if i < len(prophet_arr) else None,
            "xgboost":  float(xgb_preds[i]),
            "lightgbm": float(lgbm_preds[i]),
        }
        for i in range(n_eval)
    ]

    # Production models
    print("  [3/4] Melatih XGBoost production...")
    xgb_prod  = MultiOutputRegressor(xgb_base,  n_jobs=-1).fit(X_all, Y_v)
    print("  [4/4] Melatih LightGBM production...")
    lgbm_prod = MultiOutputRegressor(lgbm_base, n_jobs=-1).fit(X_all, Y_v)

    last_date = df["ds"].max()
    tail      = min(HIST_TAIL, len(df))
    last_rows = {s: [float(v) for v in df[s].values[-tail:]] for s in SERIES_LIST}
    historical = [
        {"date": str(r["ds"].date()), "price": float(r["silver"])}
        for _, r in df.iterrows()
    ]

    # Simpan ke disk 
    joblib.dump(xgb_prod,  os.path.join(SAVE_DIR, "xgb_prod.joblib"))
    joblib.dump(lgbm_prod, os.path.join(SAVE_DIR, "lgbm_prod.joblib"))
    with open(os.path.join(SAVE_DIR, "prophet_prod.json"), "w") as f:
        json.dump(model_to_dict(m_prod), f)

    metadata = {
        "start_date":       start_date,
        "end_date":         end_date,
        "last_actual_date": str(last_date.date()),
        "trained_at":       _dt.now().isoformat(),
        "feature_cols":     feature_cols,
        "last_rows":        last_rows,
        "test_comparison":  test_comparison,
        "historical":       historical,
        "prophet_metrics":  prophet_met,
        "xgboost_metrics":  xgboost_met,
        "lightgbm_metrics": lightgbm_met,
    }
    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    return {
        "status":           "success",
        "last_actual_date": str(last_date.date()),
        "trained_at":       metadata["trained_at"],
        "prophet":          prophet_met,
        "xgboost":          xgboost_met,
        "lightgbm":         lightgbm_met,
    }

# PREDICT FROM SAVED

def predict_from_saved(period: int = 7) -> dict:
    with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    last_rows    = meta["last_rows"]
    last_date    = pd.to_datetime(meta["last_actual_date"])

    xgb_prod  = joblib.load(os.path.join(SAVE_DIR, "xgb_prod.joblib"))
    lgbm_prod = joblib.load(os.path.join(SAVE_DIR, "lgbm_prod.joblib"))
    with open(os.path.join(SAVE_DIR, "prophet_prod.json")) as f:
        m_prod = model_from_dict(json.load(f))

    # Tanggal business days ke depan (30 days)
    future_range  = m_prod.make_future_dataframe(periods=30, freq="B")
    ph_future_all = _get_prophet_components(m_prod, future_range)
    ph_future = (
        ph_future_all[ph_future_all["ds"] > last_date]
        .head(30)
        .reset_index(drop=True)
    )

    # Komponen Prophet fitur
    ph_last = _get_prophet_components(m_prod, pd.DataFrame({"ds": [last_date]}))

    # Bangun satu feature vector di titik data terakhir
    X_pred = _feature_row(
        last_rows["silver"], last_rows["gold"],
        last_rows["oil"],    last_rows["usd"],
        float(ph_last["yhat"].iloc[0]),    float(ph_last["trend"].iloc[0]),
        float(ph_last["weekly"].iloc[0]),  float(ph_last["yearly"].iloc[0]),
        last_date, feature_cols,
    ).reshape(1, -1)

    log_rets_xgb  = xgb_prod.predict(X_pred)[0]
    log_rets_lgbm = lgbm_prod.predict(X_pred)[0]
    last_price    = float(last_rows["silver"][-1])

    # price[t+h] = last_price * exp(cumulative_log_return_h)
    future_forecast = [
        {
            "date":     str(pd.to_datetime(ph_future.loc[i, "ds"]).date()),
            "prophet":  float(ph_future.loc[i, "yhat"]),
            "xgboost":  float(last_price * np.exp(log_rets_xgb[i])),
            "lightgbm": float(last_price * np.exp(log_rets_lgbm[i])),
        }
        for i in range(min(period, len(ph_future)))
    ]

    future_actual = _fetch_future_actual(future_forecast, last_date)

    return {
        "period":           period,
        "last_actual_date": meta["last_actual_date"],
        "prophet":          meta["prophet_metrics"],
        "xgboost":          meta["xgboost_metrics"],
        "lightgbm":         meta["lightgbm_metrics"],
        "test_comparison":  meta["test_comparison"],
        "future_forecast":  future_forecast,
        "future_actual":    future_actual,
        "historical":       meta["historical"],
    }
