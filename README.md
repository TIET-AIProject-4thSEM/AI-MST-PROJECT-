# 🖥️ Proactive Thermal Management via Supervised ML

> A physics-aware machine learning system that predicts CPU temperature **5 seconds ahead** and adjusts cooling *before* critical thresholds are hit — shifting data center thermal management from reactive to proactive.

**Team Project | Cloud Infrastructure & ML**  
**Hardware:** REES52 DS18B20 Temperature Sensor + REES52 L9110 H-Bridge Fan Module

---

## The Problem

Traditional data center cooling is **reactive** — a fan ramps up only after a temperature threshold is crossed. By then the CPU has already experienced thermal stress. Cooling accounts for 30–50% of data center energy, and over-cooling (keeping fans at max "just in case") wastes a significant fraction of that.

## The Solution

A supervised ML model trained on real hardware telemetry that acts as a **trend-aware real-time state estimator**. It reads CPU load, memory usage, and ambient temperature at 1 Hz, and predicts where the CPU temperature will be in 5 seconds — giving the cooling system time to respond smoothly and efficiently.

---

## Repository Structure

```
├── cpu_temp_prediction_clean.ipynb   # Full ML pipeline (EDA → training → evaluation)
├── thermal_demo.py                   # Live 1 Hz prediction demo for presentations
├── models/
│   ├── best_thermal_model.pkl        # Trained model (Random Forest)
│   ├── feature_scaler.pkl            # StandardScaler (used by Linear Regression)
│   └── model_info.json               # Feature names, RMSE, R² metadata
└── results/
    ├── demo_log.csv                  # Per-second prediction log from demo runs
    └── demo_summary.png              # Auto-generated 4-panel summary chart
```

---

## Dataset

| Property | Value |
|---|---|
| Observations | 15,757 |
| Sample rate | 1 Hz (one reading per second) |
| Collection period | ~4.5 hours |
| Workload phases | 6 (Idle → Moderate → High → Burst → Maximum → Decay) |

**Inputs collected via `psutil`:** `cpu_utilization`, `memory_usage`, `cpu_temp`  
**Ambient temperature:** REES52 DS18B20 digital sensor (12-bit, ±0.5°C accuracy)  
**Target variable:** `cpu_temp` — CPU die temperature in °C

The six workload phases were cycled deliberately to capture both thermal *rise* (Joule heating) and *decay* (Newton's Law of Cooling), giving the model exposure to the full thermal dynamic range.

---

## Feature Engineering

The model's accuracy comes from encoding **physical reality** into the features rather than treating this as a generic regression problem.

| Feature | Physical rationale |
|---|---|
| `cpu_load_lag1 / lag5 / lag10` | Thermal inertia — heat builds with a 5–20s delay after a load spike |
| `cpu_temp_lag1 / lag5` | Temperature momentum — where it was predicts where it's going |
| `temp_rate`, `temp_acceleration` | First and second derivatives of temperature (like velocity and acceleration) |
| `cpu_load_roll10`, `cpu_temp_roll10` | 10-sample rolling mean — removes 1 Hz sensor noise without blurring trends |
| `thermal_stress` | `cpu_load × cpu_temp` — interaction term for compound heating risk |
| `temp_above_ambient` | Distance from thermal floor (Newton's Law of Cooling baseline) |
| `hour_sin`, `hour_cos` | Cyclic encoding of time-of-day — avoids ordinal artefact of raw hour integer |
| `is_high_load`, `is_heating`, `is_cooling` | Regime flags that help the model switch decision boundaries |

> ⚠️ **Anti-leakage rule:** Smoothing and lag features are applied only to input features, never to `cpu_temp`. Smoothing the target would bleed future values into training labels.

---

## Models & Results

All models were evaluated on a **chronological 80/20 holdout split** — training on the first 80% of timestamps and testing on the last 20%. A random shuffle was explicitly avoided to prevent future data leaking into training.

| Model | R² | RMSE (°C) | MAE (°C) |
|---|---|---|---|
| Linear Regression | 0.8310 | 6.058 | 4.597 |
| Decision Tree | 0.9457 | 3.435 | 2.463 |
| **Random Forest** ✅ | **0.9516** | **3.242** | **2.368** |
| Gradient Boosting | 0.9365 | 3.713 | 2.672 |

**Why Random Forest wins here:** At 1 Hz, sensor telemetry contains a lot of high-frequency noise. Gradient Boosting's sequential error-correction tends to fit individual noisy readings, while Random Forest's *bagging* approach averages those errors away — producing a smoother, more reliable predictor for real-time use.

**Recommended safety buffer:** Given an RMSE of ~3.2°C, the system uses a ±5°C safety band. Cooling is triggered at 80°C when the actual danger threshold is 85°C.

---

## Running the Live Demo

The demo runs independently of the notebook. Point it at a trained model and launch your workload generator in a separate terminal.

```bash
# default: 5-minute run, no Arduino
python thermal_demo.py

# custom duration
python thermal_demo.py --minutes 10

# with Arduino on a specific port
python thermal_demo.py --minutes 5 --port /dev/ttyUSB0   # Linux
python thermal_demo.py --minutes 5 --port COM4            # Windows
```

### What you'll see

```
  Time      Current   Predicted   Δ (5s)   Status      Fan PWM   Load
  ────────────────────────────────────────────────────────────────────
  14:22:01   58.00°C    61.34°C   +3.34°C  ELEVATED    100/255    72%
  14:22:02   58.50°C    63.10°C   +4.60°C  WARNING     108/255    85%
  14:22:03   59.20°C    65.80°C   +6.60°C  WARNING     128/255    91%
```

Colour coding: `GREEN` Normal · `BLUE` Elevated · `YELLOW` Warning · `RED` Critical

At the end the script saves:
- `results/demo_log.csv` — full per-second record of temps, predictions, fan speed, and load
- `results/demo_summary.png` — 4-panel dark-theme chart (temperature trace, CPU load, fan response)

### Hardware (optional)

The demo works in three modes and degrades gracefully:

| Mode | CPU temp source | Ambient source | Fan control |
|---|---|---|---|
| Full hardware | `psutil` sensors | DS18B20 via Arduino | L9110 PWM via Arduino |
| No Arduino | `psutil` sensors | Sine-wave simulation | Logged only |
| No sensors (VM) | Load-based estimate | Sine-wave simulation | Logged only |

---

## Running the Notebook

Open `cpu_temp_prediction_clean.ipynb` in Jupyter. Cells run top-to-bottom in order:

1. **Imports** — all dependencies loaded in one cell
2. **Load & Inspect** — shape, NaN counts, ERROR string audit
3. **EDA** — temperature trace and feature distribution plots
4. **Preprocessing** — ERROR → NaN coercion, median imputation
5. **Feature Engineering** — lags, rolling stats, cyclic time encoding
6. **Train / Test Split** — chronological 80/20
7–10. **Models** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting
11. **Results Table** — colour-gradient styled comparison
12. **Visualisations** — time-series traces, scatter plots, performance bars, feature importance, residuals
13. **Conclusions** — findings, safety margin recommendation, future directions

---

## Dependencies

```
psutil
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
pyserial        # optional — only needed for Arduino/DS18B20 hardware
```

Install with:

```bash
pip install psutil numpy pandas scikit-learn matplotlib seaborn joblib pyserial
```

---

## Hardware Setup (Optional)

```
Raspberry Pi / PC
│
├── USB → Arduino Uno
│         │
│         ├── Pin 2  ──── DS18B20 DATA  (ambient temp sensor)
│         │         └─── 4.7kΩ pull-up to 3.3V
│         │
│         ├── Pin 8  ──── L9110 IB  (direction: LOW = forward)
│         └── Pin 9  ──── L9110 IA  (PWM speed: 0–255)
│
└── L9110 ─── Fan (up to 800mA per channel)
```

Arduino sketch should respond to:
- `T\n` — return DS18B20 temperature as a float string (e.g. `24.1250`)
- `F<0-255>\n` — set fan PWM (e.g. `F128`)

---

## Key Design Decisions

**Why not lag `cpu_temp` directly?** Using temperature history as a feature would give the model nearly the full answer before it predicts. The goal is to forecast from *workload signals* that arrive before the thermal response, not to do auto-regression on temperature itself.

**Why chronological split?** In a random split, the model sees data from 14:30 in training and is tested on 14:28 — "future" data leaks into training and accuracy is artificially inflated. The chronological split mirrors actual deployment conditions.

**Why smooth inputs but not the target?** Smoothing inputs reduces noise the model would otherwise have to learn through. Smoothing the target would shift the labels toward neighboring timestamps, creating leakage.

---

## Future Directions

- **Anomaly detection** — add an Isolation Forest layer to flag deviations between predicted and actual temperature, which can indicate hardware faults (dried thermal paste, dust accumulation, failing fan)
- **Expanded features** — clock speed, supply voltage, and network I/O for a richer thermal profile
- **Time-series cross-validation** — expanding-window CV for more robust hyperparameter tuning
- **Streaming deployment** — wrap the saved model in a FastAPI service with a WebSocket endpoint for real-time dashboard integration
- **VM migration trigger** — when a server's predicted temperature exceeds the warning threshold, emit an event to the orchestration layer to begin live migration of the heaviest workloads