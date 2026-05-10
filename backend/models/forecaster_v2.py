import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
from datetime import datetime as _dt

import joblib
import pandas as pd
import numpy as np
from prophet.serialize import model_to_dict, model_from_dict

from models.forecaster import (
    fetch_data,
    build_features,
    _fit_prophet,
    _get_prophet_components,
    _feature_row,
    _build_boosting_models,
    _metrics,
    _fetch_usd_to_idr,
    _fetch_future_actual,
    SERIES_LIST,
    LAG_DAYS,
    ROLLING_WINDOWS,
    TEST_SIZE,
)

SAVE_DIR_V2 = os.path.join(os.path.dirname(__file__), "saved_v2")
HIST_TAIL   = 100


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
    tech_cols     = ["silver_rsi_14", "silver_macd", "silver_macd_signal",
                     "silver_bb_pos", "silver_atr_14"]
    ratio_cols    = ["gold_silver_ratio_lag_1", "gold_silver_ratio_lag_5",
                     "silver_oil_ratio_lag_1"]
    return price_lag_cols + ret_lag_cols + roll_cols + calendar_cols + prophet_cols + tech_cols + ratio_cols


def is_model_saved_v2() -> bool:
    required = ["xgb_prod.joblib", "lgbm_prod.joblib", "prophet_prod.json", "metadata.json"]
    return all(os.path.exists(os.path.join(SAVE_DIR_V2, f)) for f in required)


def get_model_status_v2() -> dict:
    if not is_model_saved_v2():
        return {"trained": False}
    with open(os.path.join(SAVE_DIR_V2, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "trained":          True,
        "last_actual_date": meta["last_actual_date"],
        "trained_at":       meta["trained_at"],
    }


def train_and_save_v2(start_date: str = None, end_date: str = None) -> dict:
    os.makedirs(SAVE_DIR_V2, exist_ok=True)

    raw = fetch_data(start_date, end_date)
    df  = build_features(raw)

    test_size = max(TEST_SIZE, int(len(df) * 0.2))
    train = df.iloc[:-test_size].copy().reset_index(drop=True)
    test  = df.iloc[-test_size:].copy().reset_index(drop=True)

    feature_cols = _build_feature_cols()

    #  Evaluation model (train set only) 
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

    #  Rolling origin evaluation on test set 
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

    xgboost_met  = _metrics(actual_test, xgb_final)
    lightgbm_met = _metrics(actual_test, lgbm_final)

    test_comparison = [
        {
            "date":     str(test["ds"].iloc[i].date()),
            "actual":   float(actual_test[i]),
            "xgboost":  float(xgb_final[i]),
            "lightgbm": float(lgbm_final[i]),
        }
        for i in range(TEST_SIZE)
    ]

    #  Production model (all data) 
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

    last_date = df["ds"].max()
    tail      = min(HIST_TAIL, len(df))
    last_rows = {s: [float(v) for v in df[s].values[-tail:]] for s in SERIES_LIST}
    historical = [
        {"date": str(row["ds"].date()), "price": float(row["silver"])}
        for _, row in df.iterrows()
    ]

    #  Save to disk 
    joblib.dump(xgb_prod,  os.path.join(SAVE_DIR_V2, "xgb_prod.joblib"))
    joblib.dump(lgbm_prod, os.path.join(SAVE_DIR_V2, "lgbm_prod.joblib"))
    with open(os.path.join(SAVE_DIR_V2, "prophet_prod.json"), "w", encoding="utf-8") as f:
        json.dump(model_to_dict(m_prod), f)

    trained_at = _dt.now().isoformat()
    metadata = {
        "start_date":       start_date,
        "end_date":         end_date,
        "last_actual_date": str(last_date.date()),
        "trained_at":       trained_at,
        "feature_cols":     feature_cols,
        "last_rows":        last_rows,
        "test_comparison":  test_comparison,
        "historical":       historical,
        "xgboost_metrics":  xgboost_met,
        "lightgbm_metrics": lightgbm_met,
    }
    with open(os.path.join(SAVE_DIR_V2, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return {
        "status":           "success",
        "last_actual_date": str(last_date.date()),
        "trained_at":       trained_at,
        "xgboost":          xgboost_met,
        "lightgbm":         lightgbm_met,
    }


def predict_from_saved_v2(period: int = 7, end_date: str = None) -> dict:
    with open(os.path.join(SAVE_DIR_V2, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]

    xgb_prod  = joblib.load(os.path.join(SAVE_DIR_V2, "xgb_prod.joblib"))
    lgbm_prod = joblib.load(os.path.join(SAVE_DIR_V2, "lgbm_prod.joblib"))
    with open(os.path.join(SAVE_DIR_V2, "prophet_prod.json"), encoding="utf-8") as f:
        m_prod = model_from_dict(json.load(f))

    if end_date:
        raw = fetch_data(end_date=end_date)
        df  = build_features(raw)
        ph  = _get_prophet_components(m_prod, df[["ds"]].copy())
        for col in ["yhat", "trend", "weekly", "yearly"]:
            df[f"ph_{col}"] = ph[col].values

        last_date   = df["ds"].max()
        silver_hist = list(df["silver"].values)
        gold_hist   = list(df["gold"].values)
        oil_hist    = list(df["oil"].values)
        usd_hist    = list(df["usd"].values)
        historical  = [
            {"date": str(row["ds"].date()), "price": float(row["silver"])}
            for _, row in df.iterrows()
        ]
    else:
        last_date   = pd.to_datetime(meta["last_actual_date"])
        silver_hist = meta["last_rows"]["silver"]
        gold_hist   = meta["last_rows"]["gold"]
        oil_hist    = meta["last_rows"]["oil"]
        usd_hist    = meta["last_rows"]["usd"]
        historical  = meta["historical"]

    # Generate future business days
    future_range  = m_prod.make_future_dataframe(periods=period, freq="B")
    ph_future_all = _get_prophet_components(m_prod, future_range)
    ph_future = (
        ph_future_all[ph_future_all["ds"] > last_date]
        .head(period)
        .reset_index(drop=True)
    )

    # Iterative prediction
    future_silver_hist_xgb  = list(silver_hist)
    future_silver_hist_lgbm = list(silver_hist)
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

    future_forecast = [
        {
            "date":     str(pd.to_datetime(ph_future.loc[i, "ds"]).date()),
            "xgboost":  float(xgb_future_vals[i]),
            "lightgbm": float(lgbm_future_vals[i]),
        }
        for i in range(len(ph_future))
    ]

    future_actual = _fetch_future_actual(future_forecast, last_date)
    usd_to_idr    = _fetch_usd_to_idr()

    return {
        "period":           period,
        "last_actual_date": str(last_date.date()),
        "xgboost":          meta["xgboost_metrics"],
        "lightgbm":         meta["lightgbm_metrics"],
        "test_comparison":  meta["test_comparison"],
        "future_forecast":  future_forecast,
        "future_actual":    future_actual,
        "historical":       historical,
        "usd_to_idr":       usd_to_idr,
    }
