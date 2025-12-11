import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("uae_er_preprocessed.csv")

# These are the ONLY features the dashboard uses
FEATURES = [
    'hospital_id', 'is_weekend', 'is_public_holiday', 'month', 'day_of_year',
    'temperature_C', 'humidity_pct', 'expected_visits', 'critical_cases',
    'prev_day_visits', 'ma_7', 'ma_14', 'dow_num'
]

TARGET = "daily_ER_visits"

# Select exact columns
df = df[FEATURES + [TARGET]]

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = XGBRegressor(
    n_estimators=350,
    learning_rate=0.06,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training Features:", X_train.columns.tolist())

model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Model MAE:", mae)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
joblib.dump(model, "xgb_model.joblib")
print("Model saved as xgb_model.joblib")
