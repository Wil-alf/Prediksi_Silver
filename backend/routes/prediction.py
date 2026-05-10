from flask import Blueprint, request, jsonify
from models.forecaster import run_forecast
from datetime import datetime, timedelta
import yfinance as yf

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    body     = request.get_json(silent=True) or {}
    period   = body.get("period", 7)
    end_date = body.get("end_date")

    if period not in (7, 30):
        return jsonify({"error": "period harus 7 atau 30"}), 400

    try:
        result = run_forecast(period=int(period), end_date=end_date)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route("/historical", methods=["GET"])
def historical():
    start_date = request.args.get("start_date")
    end_date   = request.args.get("end_date")
    try:
        kwargs: dict = {"interval": "1d", "auto_adjust": True, "progress": False}
        if start_date and end_date:
            end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            kwargs["start"] = start_date
            kwargs["end"]   = end_dt
        else:
            kwargs["period"] = "10y"
        raw = yf.download("SI=F", **kwargs)
        if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
            close = raw["Close"]
        else:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        close = close.dropna().reset_index()
        close.columns = ["date", "price"]
        close["date"]  = close["date"].astype(str).str[:10]
        close["price"] = close["price"].astype(float)
        return jsonify(close.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
