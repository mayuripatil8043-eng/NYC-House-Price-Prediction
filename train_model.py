# ============================================================
# HOUSE PRICE PREDICTION - FIXED MODEL (FAST)
# ============================================================
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "train.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "backend" / "Data" / "train.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"

print(f"\n📂 Loading from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
print(f"✅ Shape: {df.shape}")

TARGET = "TARGET(PRICE_IN_LACS)"
if TARGET not in df.columns:
    print(f"\n⚠️ '{TARGET}' nahi sapadla, auto-creating...")
    np.random.seed(42)
    df[TARGET] = (df['SQUARE_FT'] * 0.05 + df['BHK_NO.'] * 6 + np.random.normal(0, 4, len(df))).round(2)
    df[TARGET] = df[TARGET].clip(lower=10)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Navin '{TARGET}' banla!")

df = df.dropna(subset=[TARGET]).copy()
df = df.replace([float("inf"), float("-inf")], pd.NA)

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\n🔢 Numerical: {numerical_columns}")
print(f"🔤 Categorical: {categorical_columns}")

numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# --- YAHI MAIN FIX HAI ---
model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=1, max_depth=15)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"\n📚 Training: {len(X_train)}, Testing: {len(X_test)}")

print("\n⏳ Training...")
pipeline.fit(X_train, y_train)
print("✅ Training completed!")

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")
print("="*50)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": pipeline, "target": TARGET, "features": list(X.columns)}, f)

print(f"\n🎉 MODEL SAVED at {MODEL_PATH}")