from flask import Flask, request, render_template, jsonify
from pathlib import Path
import pickle
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = PROJECT_DIR / "model" / "house_price_model.pkl"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)


# ============================================================
# LOAD MODEL
# ============================================================

model = None

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("✅ Model loaded successfully")
    print(f"📦 Model: {MODEL_PATH}")

except Exception as e:
    print("❌ Model loading error:")
    print(e)


# ============================================================
# HOME / PREDICTION
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            # Get form data
            data = request.form.to_dict()

            print("\n📥 Received data:")
            print(data)

            # Check empty input
            if not data:
                error = "No input data received."

                return render_template(
                    "index.html",
                    prediction=prediction,
                    error=error
                )

            # ------------------------------------------------
            # Create DataFrame
            # ------------------------------------------------

            input_data = pd.DataFrame([data])

            # ------------------------------------------------
            # Numeric columns
            # ------------------------------------------------

            numeric_columns = [
                "UNDER_CONSTRUCTION",
                "RERA",
                "BHK_NO.",
                "SQUARE_FT",
                "READY_TO_MOVE",
                "RESALE",
                "LONGITUDE",
                "LATITUDE"
            ]

            for column in numeric_columns:

                if column in input_data.columns:

                    input_data[column] = pd.to_numeric(
                        input_data[column],
                        errors="coerce"
                    )

            # ------------------------------------------------
            # Check model
            # ------------------------------------------------

            if model is None:

                error = (
                    "Model file could not be loaded. "
                    "Please check house_price_model.pkl."
                )

            else:

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                result = model.predict(input_data)[0]

                prediction = round(float(result), 2)

                print(f"🏠 Predicted Price: {prediction}")

        except Exception as e:

            print("❌ Prediction error:")
            print(e)

            error = f"Prediction error: {str(e)}"

    # --------------------------------------------------------
    # Render page
    # --------------------------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


# ============================================================
# OPTIONAL API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400

        input_data = pd.DataFrame([data])

        numeric_columns = [
            "UNDER_CONSTRUCTION",
            "RERA",
            "BHK_NO.",
            "SQUARE_FT",
            "READY_TO_MOVE",
            "RESALE",
            "LONGITUDE",
            "LATITUDE"
        ]

        for column in numeric_columns:

            if column in input_data.columns:

                input_data[column] = pd.to_numeric(
                    input_data[column],
                    errors="coerce"
                )

        if model is None:

            return jsonify({
                "success": False,
                "error": "Model is not loaded."
            }), 500

        result = model.predict(input_data)[0]

        prediction = round(float(result), 2)

        return jsonify({
            "success": True,
            "predicted_price": prediction
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🏠 HOUSE PRICE PREDICTION")
    print("=" * 60)

    print(f"📁 Project:  {PROJECT_DIR}")
    print(f"📁 Template: {TEMPLATE_DIR}")
    print(f"📦 Model:    {MODEL_PATH}")

    template_exists = (
        TEMPLATE_DIR / "index.html"
    ).exists()

    model_exists = MODEL_PATH.exists()

    print(f"📄 index.html exists: {template_exists}")
    print(f"📦 Model exists:      {model_exists}")

    print("=" * 60)

    if not template_exists:

        print("❌ ERROR: index.html not found!")
        print("👉 Put index.html inside:")
        print(f"   {TEMPLATE_DIR}")

    elif not model_exists:

        print("❌ ERROR: Model file not found!")
        print("👉 Check:")
        print(f"   {MODEL_PATH}")

    else:

        print("✅ Project files detected.")
        print("🚀 Starting Flask server...")
        print("🌐 http://127.0.0.1:5000")

    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
