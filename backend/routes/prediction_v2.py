from flask import Blueprint, jsonify, request
from models.forecaster_v2 import (
    get_model_status_v2,
    train_and_save_v2,
    predict_from_saved_v2,
    is_model_saved_v2,
)

prediction_v2_bp = Blueprint("prediction_v2", __name__)


@prediction_v2_bp.route("/model-status", methods=["GET"])
def model_status():
    return jsonify(get_model_status_v2())


@prediction_v2_bp.route("/train", methods=["POST"])
def train():
    body     = request.get_json(silent=True) or {}
    end_date = body.get("end_date")
    try:
        result = train_and_save_v2(end_date=end_date)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_v2_bp.route("/predict", methods=["POST"])
def predict():
    if not is_model_saved_v2():
        return jsonify({"error": "Model belum dilatih. Jalankan /api/v2/train terlebih dahulu."}), 503

    body     = request.get_json(silent=True) or {}
    period   = body.get("period", 7)
    end_date = body.get("end_date")

    if period not in (7, 30):
        return jsonify({"error": "period harus 7 atau 30"}), 400

    try:
        result = predict_from_saved_v2(period=int(period), end_date=end_date)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
