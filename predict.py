import joblib
import pandas as pd

model = joblib.load("cpu_temp_predictor.pkl")
feature_cols = joblib.load("feature_columns.pkl")

print("Model loaded successfully")

sample = {
    "cpu_load": 25,
    "ram_usage": 35,
    "ambient_temp": 22,
    "cpu_temp_lag1": 65,
    "cpu_temp_lag5": 63,
    "cpu_load_lag1": 24,
    "cpu_load_lag5": 20,
    "cpu_load_lag10": 18,
    "temp_rate": 0.2,
    "temp_accelaration": 0.05,
    "load_rate": 0.1,
    "cpu_temp_roll10": 64,
    "cpu_load_roll10": 22,
    "cpu_load_roll30": 20,
    "cpu_load_std10": 1.5,
    "load_ambient_interaction": 550,
    "thermal_stress": 1625,
    "temp_above_ambient": 42,
}

input_df = pd.DataFrame([sample])
input_df = input_df[feature_cols]

prediction = model.predict(input_df)[0]
print("Predicted CPU temperature (5 seconds ahead):", prediction)

THRESHOLD = 75

if prediction > THRESHOLD:
    print("TURN FAN ON")
else:
    print("FAN OFF")
