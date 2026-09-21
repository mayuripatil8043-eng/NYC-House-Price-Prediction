from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pickle
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "model" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "Data" / "train.csv"

# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

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
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# If features were not saved inside pickle
if not features:

    target_columns = [
        "price",
        "medv",
        "target",
        "price_in_lacs",
        "TARGET(PRICE_IN_LACS)"
    ]

    features = [
        col for col in df.columns
        if col.lower() not in [
            x.lower() for x in target_columns
        ]
    ]

features = list(features)

# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = []

for feature in features:

    if feature in df.columns:

        if pd.api.types.is_numeric_dtype(df[feature]):
            numeric_features.append(feature)

# ============================================================
# DEBUG INFORMATION
# ============================================================

print("\n==========================================")
print("HOUSE PRICE PREDICTION")
print("==========================================")
print("Template:", TEMPLATE_DIR)
print("Index:", TEMPLATE_DIR / "index.html")
print("Index exists:", (TEMPLATE_DIR / "index.html").exists())
print("\nFeatures:")

for feature in features:
    print("-", feature)

print("==========================================\n")


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        features=features,
        numeric_features=numeric_features
    )


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Accept normal HTML form OR JSON
        # ----------------------------------------------------

        if request.is_json:

            data = request.get_json(silent=True)

        else:

            data = request.form.to_dict()

        # ----------------------------------------------------
        # Check data
        # ----------------------------------------------------

        if not data:

            return jsonify({
                "success": False,
                "error": "Please enter house details."
            }), 400

        # ----------------------------------------------------
        # Missing fields
        # ----------------------------------------------------

        missing = []

        for feature in features:

            if feature not in data:
                missing.append(feature)

        if missing:

            return jsonify({
                "success": False,
                "error": "Please fill all required fields.",
                "missing": missing
            }), 400

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        input_data = {}

        for feature in features:

            value = data.get(feature, "")

            if str(value).strip() == "":

                return jsonify({
                    "success": False,
                    "error": f"Please enter {feature}."
                }), 400

            # Numeric columns
            if feature in numeric_features:

                input_data[feature] = [
                    float(value)
                ]

            # Text columns
            else:

                input_data[feature] = [
                    str(value).strip()
                ]

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(input_data)

        # Keep same order as training
        input_df = input_df[features]

        print("\nInput received:")
        print(input_df)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_df)

        price = float(prediction[0])

        print("Prediction:", price)

        return jsonify({
            "success": True,
            "predicted_price": round(price, 2)
        })

    except Exception as error:

        print("\nPREDICTION ERROR:")
        print(error)

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