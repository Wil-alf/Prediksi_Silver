import os
from flask import Flask
from flask_cors import CORS
from routes.prediction import prediction_bp
from routes.prediction_v2 import prediction_v2_bp

app = Flask(__name__)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
CORS(app, origins=allowed_origins)

app.register_blueprint(prediction_bp, url_prefix="/api")
app.register_blueprint(prediction_v2_bp, url_prefix="/api/v2")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")