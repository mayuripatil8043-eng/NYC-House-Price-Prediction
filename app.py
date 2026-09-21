from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pickle
import pandas as pd

# ============================================================
# HOUSE PRICE PREDICTION - FLASK BACKEND
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Flask ko explicitly templates folder bata rahe hain
TEMPLATE_DIR = CURRENT_DIR / "templates"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = CURRENT_DIR.parent

MODEL_PATH = BASE_DIR / "model" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "Data" / "train.csv"

# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:
    saved = pickle.load(file)

if isinstance(saved, dict):
    model = saved["model"]
    features = saved.get("features")
else:
    model = saved
    features = None

# ============================================================
# LOAD FEATURES
# ============================================================

if not features:
    df = pd.read_csv(DATA_PATH)

    features = [
        column
        for column in df.columns
        if column.lower() != "medv"
    ]

features = list(features)

print("Model loaded successfully.")
print("Templates folder:", TEMPLATE_DIR)
print("Template exists:", (TEMPLATE_DIR / "index.html").exists())
print("Features:", features)

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        features=features
    )

# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400

        missing = [
            feature
            for feature in features
            if feature not in data
        ]

        if missing:
            return jsonify({
                "success": False,
                "error": "Missing features.",
                "missing_features": missing
            }), 400

        input_data = {}

        for feature in features:
            input_data[feature] = [
                float(data[feature])
            ]

        input_df = pd.DataFrame(input_data)

        prediction = model.predict(input_df)

        return jsonify({
            "success": True,
            "predicted_price": round(float(prediction[0]), 2)
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )