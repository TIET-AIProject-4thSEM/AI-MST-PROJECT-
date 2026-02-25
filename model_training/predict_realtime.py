"""
Proactive Thermal Management — Live Demo
=========================================
Run this while your workload generator stresses the CPU.

HOW THE EARLY WARNING WORKS
────────────────────────────
The model predicts the *current* CPU temperature from workload signals
(CPU utilisation, memory usage, ambient temp and their lags/smoothed values).

Because of thermal inertia, CPU load spikes 5–20 seconds BEFORE the die
temperature catches up (Newton's Law of Cooling / Joule heating delay).
So when the model returns a predicted_temp that is higher than the actual
current reading, it means the load has already spiked but the hardware
hasn't heated up yet — that gap is our proactive cooling window.

  predicted_temp > current_temp  →  heating trajectory detected → ramp fan NOW
  predicted_temp ≈ current_temp  →  stable state
  predicted_temp < current_temp  →  system is cooling → can reduce fan

Usage:
    python thermal_demo.py               # 5-minute run
    python thermal_demo.py --minutes 10  # custom duration
    python thermal_demo.py --port COM4   # with Arduino on a specific port

Hardware (optional): REES52 DS18B20 sensor + REES52 L9110 H-bridge fan module.
If no Arduino is found, ambient temperature is simulated and fan commands are
logged only (no hardware output).
"""

import psutil
import time
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import argparse
import warnings
from datetime import datetime
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── optional Arduino ──────────────────────────────────────────────────────────
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = 'models/best_thermal_model.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'
INFO_PATH   = 'models/model_info.json'
RESULT_CSV  = 'results/demo_log.csv'
RESULT_PLOT = 'results/demo_summary.png'

# ── thresholds ────────────────────────────────────────────────────────────────
TEMP_WARNING  = 70.0   # °C — begin ramping fan
TEMP_CRITICAL = 80.0   # °C — maximum fan
SAFETY_BUFFER = 5.0    # °C — safety margin on top of model RMSE (~3.2°C)

# samples needed before the first prediction (lag-6 needs index -7, +1 guard)
WARMUP_SAMPLES = 8


# ═════════════════════════════════════════════════════════════════════════════
class ThermalDemo:
    """
    1 Hz proactive thermal prediction loop.

    Features match exactly what the training notebook produced:
        cpu_utilization_smooth, memory_usage_smooth, ambient_temp_smooth,
        hour, day_of_week, is_business,
        cpu_util_lag1, cpu_util_lag3, cpu_util_lag6
    """

    def __init__(self, arduino_port=None):
        self.model         = None
        self.scaler        = None
        self.feature_names = None
        self.model_rmse    = 3.24   # fallback if not in model_info.json

        self.arduino    = None
        self.arduino_ok = False

        self.history = deque(maxlen=30)
        self.log     = []

        self.last_fan_pwm = 0
        self.max_fan_step = 20

        self._load_model()
        if arduino_port:
            self._init_arduino(arduino_port)

        psutil.cpu_percent(interval=None)   # prime non-blocking call
        time.sleep(0.1)

    # ── model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        for path in (MODEL_PATH, SCALER_PATH):
            if not os.path.exists(path):
                print(f"{RED}✗ Required file not found: {path}{RESET}")
                print("  Run Section 12 of the training notebook first.")
                sys.exit(1)

        self.model  = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        if os.path.exists(INFO_PATH):
            with open(INFO_PATH) as f:
                info = json.load(f)
            self.feature_names = info['features']
            self.model_rmse    = info.get('test_rmse', self.model_rmse)
            print(f"{GREEN}✓ Model loaded{RESET}  — {info['model_name']}")
            print(f"  Test RMSE : {info['test_rmse']:.3f}°C   |   Test R² : {info['test_r2']:.4f}")
            print(f"  Features  : {self.feature_names}")
        else:
            print(f"{YELLOW}⚠  model_info.json not found — feature order inferred from model.{RESET}")

    # ── Arduino / DS18B20 ─────────────────────────────────────────────────────
    def _init_arduino(self, port):
        if not SERIAL_AVAILABLE:
            print("⚠  pyserial not installed — running without hardware.")
            return

        for p in [port, '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', 'COM3', 'COM4', 'COM5']:
            try:
                self.arduino = serial.Serial(p, 9600, timeout=1)
                time.sleep(2.5)
                self.arduino.reset_input_buffer()
                self.arduino.reset_output_buffer()
                self.arduino.write(b'T\n')
                time.sleep(0.85)
                if self.arduino.in_waiting:
                    raw  = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    temp = float(raw)
                    if -55 <= temp <= 125:
                        print(f"{GREEN}✓ DS18B20 on {p}{RESET}  — {temp:.4f}°C")
                        self.arduino_ok = True
                        return
            except Exception:
                continue

        print("⚠  Arduino not found — ambient temperature will be simulated.")

    # ── sensors ───────────────────────────────────────────────────────────────
    def _read_cpu_temp(self):
        try:
            sensors = psutil.sensors_temperatures()
            for key in ('coretemp', 'k10temp', 'cpu_thermal'):
                if key in sensors:
                    return sensors[key][0].current
            return list(sensors.values())[0][0].current
        except Exception:
            load = psutil.cpu_percent(interval=None)
            return 35.0 + load * 0.4 + np.random.normal(0, 1.0)

    def _read_ambient(self):
        if self.arduino_ok:
            try:
                self.arduino.reset_input_buffer()
                self.arduino.write(b'T\n')
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1.0:
                    if self.arduino.in_waiting:
                        raw = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                        v   = float(raw)
                        if -55 <= v <= 125:
                            return v
                    time.sleep(0.01)
                self.arduino_ok = False
            except Exception:
                self.arduino_ok = False
        return 24.0 + 2.0 * np.sin(time.time() / 3600)

    def _snapshot(self):
        return {
            'ts':       time.time(),
            'cpu_util': psutil.cpu_percent(interval=None),
            'memory':   psutil.virtual_memory().percent,
            'ambient':  self._read_ambient(),
            'cpu_temp': self._read_cpu_temp(),
        }

    # ── feature engineering ───────────────────────────────────────────────────
    def _build_features(self, snap):
        """
        Produces exactly the 9 features the notebook trained on:

            cpu_utilization_smooth  5-sample rolling mean of cpu_utilization
            memory_usage_smooth     5-sample rolling mean of memory_usage
            ambient_temp_smooth     5-sample rolling mean of ambient_temp
            hour                    raw hour of day (0-23)
            day_of_week             Monday=0 … Sunday=6
            is_business             1 if hour 8-18 on a weekday, else 0
            cpu_util_lag1           cpu_utilization 1 sample ago
            cpu_util_lag3           cpu_utilization 3 samples ago
            cpu_util_lag6           cpu_utilization 6 samples ago
        """
        self.history.append(snap)
        n = len(self.history)

        if n < WARMUP_SAMPLES:
            return None

        h = list(self.history)   # oldest → newest, h[-1] is current snap

        # rolling means — window=5, min_periods=1 (mirrors pandas training code)
        w              = min(5, n)
        cpu_smooth     = np.mean([s['cpu_util'] for s in h[-w:]])
        memory_smooth  = np.mean([s['memory']   for s in h[-w:]])
        ambient_smooth = np.mean([s['ambient']  for s in h[-w:]])

        # lag features: h[-2]=1s ago, h[-4]=3s ago, h[-7]=6s ago
        lag1 = h[-2]['cpu_util']
        lag3 = h[-4]['cpu_util']
        lag6 = h[-7]['cpu_util']

        # time features
        now         = datetime.now()
        hour        = now.hour
        dow         = now.weekday()
        is_business = int(8 <= hour <= 18 and dow < 5)

        return {
            'cpu_utilization_smooth': cpu_smooth,
            'memory_usage_smooth':    memory_smooth,
            'ambient_temp_smooth':    ambient_smooth,
            'hour':                   hour,
            'day_of_week':            dow,
            'is_business':            is_business,
            'cpu_util_lag1':          lag1,
            'cpu_util_lag3':          lag3,
            'cpu_util_lag6':          lag6,
        }

    # ── prediction ────────────────────────────────────────────────────────────
    def _predict(self, features):
        df = pd.DataFrame([features])

        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                print(f"{RED}✗ Feature mismatch — missing: {missing}{RESET}")
                print("  Ensure model_info.json matches _build_features().")
                return None
            df = df[self.feature_names]   # enforce training column order

        try:
            return float(self.model.predict(df)[0])
        except Exception:
            try:
                return float(self.model.predict(self.scaler.transform(df))[0])
            except Exception as e:
                print(f"{RED}✗ Prediction error: {e}{RESET}")
                return None

    # ── fan control ───────────────────────────────────────────────────────────
    def _fan_command(self, predicted_temp, current_temp):
        t = predicted_temp if predicted_temp is not None else current_temp

        if t >= TEMP_CRITICAL:
            target, label, colour = 255, "CRITICAL", RED
        elif t >= TEMP_WARNING:
            ratio  = (t - TEMP_WARNING) / (TEMP_CRITICAL - TEMP_WARNING)
            target = int(128 + 127 * ratio)
            label, colour = "WARNING ", YELLOW
        elif t >= 60:
            target, label, colour = 100, "ELEVATED", BLUE
        else:
            target, label, colour = 50,  "NORMAL  ", GREEN

        pwm = int(np.clip(target,
                          self.last_fan_pwm - self.max_fan_step,
                          self.last_fan_pwm + self.max_fan_step))
        self.last_fan_pwm = pwm

        if self.arduino_ok:
            try:
                self.arduino.reset_output_buffer()
                self.arduino.write(f'F{pwm}\n'.encode())
            except Exception:
                self.arduino_ok = False

        return pwm, label, colour

    # ── summary plot ──────────────────────────────────────────────────────────
    def _save_plot(self):
        os.makedirs(os.path.dirname(RESULT_PLOT) or '.', exist_ok=True)
        df = pd.DataFrame(self.log)
        df['t'] = range(len(df))

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0d1117')
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3)

        def _style(ax, title):
            ax.set_facecolor('#161b22')
            ax.set_title(title, color='white', fontsize=11, pad=8)
            ax.tick_params(colors='#8b949e')
            ax.spines[:].set_color('#30363d')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.grid(True, color='#21262d', linewidth=0.8)

        # temperature trace
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['t'], df['actual_temp'],    color='#58a6ff', lw=1.5,
                 label='Actual CPU temp', alpha=0.95)
        ax1.plot(df['t'], df['predicted_temp'], color='#f78166', lw=1.2,
                 ls='--', label='Model estimate (load-aware)', alpha=0.9)
        ax1.fill_between(df['t'],
                         df['predicted_temp'] - SAFETY_BUFFER,
                         df['predicted_temp'] + SAFETY_BUFFER,
                         color='#f78166', alpha=0.08,
                         label=f'±{SAFETY_BUFFER}°C safety band')
        ax1.axhline(TEMP_WARNING,  color='#e3b341', ls=':', lw=1.2, alpha=0.8,
                    label=f'Warning {TEMP_WARNING}°C')
        ax1.axhline(TEMP_CRITICAL, color='#f85149', ls=':', lw=1.2, alpha=0.8,
                    label=f'Critical {TEMP_CRITICAL}°C')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_xlabel('Sample (seconds)')
        ax1.legend(fontsize=8.5, loc='upper left',
                   facecolor='#1c2128', edgecolor='#30363d', labelcolor='white')
        _style(ax1, 'CPU Temperature — Actual vs Model Estimate')

        # CPU load
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df['t'], df['cpu_load'], color='#3fb950', lw=1.2, alpha=0.9)
        ax2.fill_between(df['t'], 0, df['cpu_load'], color='#3fb950', alpha=0.18)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel('CPU Utilisation (%)')
        ax2.set_xlabel('Sample (seconds)')
        _style(ax2, 'CPU Utilisation')

        # fan speed
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(df['t'], df['fan_pwm'], color='#d2a8ff', lw=1.2, alpha=0.9)
        ax3.fill_between(df['t'], 0, df['fan_pwm'], color='#d2a8ff', alpha=0.18)
        ax3.set_ylim(0, 275)
        ax3.set_ylabel('Fan PWM (0–255)')
        ax3.set_xlabel('Sample (seconds)')
        _style(ax3, 'L9110 Fan Speed (Proactive Response)')

        delta = df['predicted_temp'] - df['actual_temp']
        stats = (
            f"Samples      : {len(df)}\n"
            f"Actual range : {df['actual_temp'].min():.1f}–{df['actual_temp'].max():.1f}°C\n"
            f"Mean Δ(est−act) : {delta.mean():+.2f}°C\n"
            f"Max  Δ(est−act) : {delta.abs().max():.2f}°C\n"
            f"Ambient avg  : {df['ambient_temp'].mean():.1f}°C\n"
            f"Fan range    : {df['fan_pwm'].min()}–{df['fan_pwm'].max()} PWM"
        )
        fig.text(0.77, 0.09, stats, fontsize=8.5, color='#c9d1d9',
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='#1c2128', edgecolor='#30363d'))
        fig.suptitle('Proactive Thermal Management — Demo Summary',
                     color='white', fontsize=14, fontweight='bold', y=0.98)

        plt.savefig(RESULT_PLOT, dpi=140, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"{GREEN}✓ Plot saved → {RESULT_PLOT}{RESET}")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self, duration_minutes=5):
        hw = "DS18B20 + L9110" if self.arduino_ok else "Software simulation"
        print(f"\n{BOLD}{'═'*72}{RESET}")
        print(f"{BOLD}  PROACTIVE THERMAL MANAGEMENT — LIVE DEMO{RESET}")
        print(f"  Hardware   : {hw}")
        print(f"  Duration   : {duration_minutes} min  |  Sample rate: 1 Hz")
        print(f"  Thresholds : WARNING {TEMP_WARNING}°C  |  CRITICAL {TEMP_CRITICAL}°C")
        print(f"  Safety band: ±{SAFETY_BUFFER}°C  |  Model RMSE: ~{self.model_rmse:.2f}°C")
        print(f"\n  HOW IT WORKS: CPU load spikes before temperature does (thermal inertia).")
        print(f"  The model estimates temp from load — a +ve gap means heat is building.")
        print(f"  Fan adjusts on the ESTIMATE, not the actual reading.")
        print(f"{BOLD}{'═'*72}{RESET}\n")
        print(f"  Warming up — collecting {WARMUP_SAMPLES} initial samples…")

        os.makedirs(os.path.dirname(RESULT_CSV) or '.', exist_ok=True)

        end_time       = time.monotonic() + duration_minutes * 60
        next_tick      = time.monotonic()
        header_printed = False
        sample_n       = 0

        try:
            while time.monotonic() < end_time:
                snap     = self._snapshot()
                sample_n += 1
                features = self._build_features(snap)

                if features is None:
                    n   = len(self.history)
                    bar = '█' * n + '░' * (WARMUP_SAMPLES - n)
                    print(f"\r  [{bar}] {n}/{WARMUP_SAMPLES}", end='', flush=True)
                    next_tick += 1.0
                    _sleep_until(next_tick)
                    continue

                if not header_printed:
                    print(f"\n\n{'─'*76}")
                    print(f"  {'Time':8s}  {'Actual':>8s}  {'Estimate':>10s}  "
                          f"{'Δ(est-act)':>11s}  {'Status':9s}  {'Fan':>7s}  {'Load':>5s}")
                    print(f"{'─'*76}")
                    header_printed = True

                predicted = self._predict(features)
                if predicted is None:
                    next_tick += 1.0
                    _sleep_until(next_tick)
                    continue

                actual = snap['cpu_temp']
                delta  = predicted - actual
                pwm, label, colour = self._fan_command(predicted, actual)
                ts = datetime.now().strftime('%H:%M:%S')

                print(
                    f"  {ts}  "
                    f"{actual:6.2f}°C  "
                    f"{predicted:8.2f}°C  "
                    f"{delta:+9.2f}°C  "
                    f"{colour}{label}{RESET}  "
                    f"{pwm:4d}/255  "
                    f"{snap['cpu_util']:4.0f}%",
                    flush=True
                )

                self.log.append({
                    'timestamp':      ts,
                    'actual_temp':    actual,
                    'predicted_temp': predicted,
                    'delta':          delta,
                    'cpu_load':       snap['cpu_util'],
                    'ambient_temp':   snap['ambient'],
                    'fan_pwm':        pwm,
                    'status':         label.strip(),
                })

                next_tick += 1.0
                lag = next_tick - time.monotonic()
                if lag < -0.15:
                    print(f"  {YELLOW}⚠  sample {sample_n} lagged {-lag:.2f}s{RESET}")
                _sleep_until(next_tick)

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}⚠  Stopped by user.{RESET}")
        finally:
            self._shutdown()

    # ── shutdown ──────────────────────────────────────────────────────────────
    def _shutdown(self):
        if self.arduino_ok and self.arduino:
            try:
                self.arduino.write(b'F0\n')
                time.sleep(0.1)
                self.arduino.close()
            except Exception:
                pass

        if not self.log:
            print("No predictions were made.")
            return

        df         = pd.DataFrame(self.log)
        df.to_csv(RESULT_CSV, index=False)

        delta      = df['predicted_temp'] - df['actual_temp']
        n_warning  = (df['status'] == 'WARNING').sum()
        n_critical = (df['status'] == 'CRITICAL').sum()

        print(f"\n{BOLD}{'═'*72}{RESET}")
        print(f"{BOLD}  DEMO SUMMARY{RESET}")
        print(f"{'─'*72}")
        print(f"  {'Total predictions':32s}: {len(df)}")
        print(f"  {'Actual temp range':32s}: {df['actual_temp'].min():.1f}°C – {df['actual_temp'].max():.1f}°C")
        print(f"  {'Mean Δ (estimate − actual)':32s}: {delta.mean():+.2f}°C")
        print(f"  {'Max  Δ (estimate − actual)':32s}: {delta.abs().max():.2f}°C")
        print(f"  {'Ambient avg (DS18B20/sim)':32s}: {df['ambient_temp'].mean():.2f}°C")
        print(f"  {'Fan PWM range':32s}: {df['fan_pwm'].min()} – {df['fan_pwm'].max()} / 255")
        print(f"  {'WARNING-level ticks':32s}: {n_warning}")
        print(f"  {'CRITICAL-level ticks':32s}: {n_critical}")
        print(f"{'─'*72}")
        print(f"  {GREEN}✓ Log saved → {RESULT_CSV}{RESET}")

        self._save_plot()
        print(f"{BOLD}{'═'*72}{RESET}\n")


# ── helpers ───────────────────────────────────────────────────────────────────
def _sleep_until(t):
    remaining = t - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"""
\033[1m\033[96m╔══════════════════════════════════════════════════════════════╗
║       PROACTIVE THERMAL MANAGEMENT  ·  LIVE DEMO            ║
║  Load-aware ML estimate → fan acts before temperature spikes ║
╚══════════════════════════════════════════════════════════════╝\033[0m
""")
    parser = argparse.ArgumentParser(description='Thermal prediction live demo')
    parser.add_argument('--minutes', type=int, default=5,
                        help='Monitoring duration in minutes (default: 5)')
    parser.add_argument('--port', type=str, default=None,
                        help='Arduino serial port, e.g. COM4 or /dev/ttyUSB0')
    args = parser.parse_args()

    demo = ThermalDemo(arduino_port=args.port)
    print(f"\n\033[1m▶ Starting {args.minutes}-minute demo…\033[0m")
    print("  Launch your workload generator now, then watch the output.")
    print("  Ctrl+C stops early and still saves the summary.\n")
    time.sleep(2)

    demo.run(duration_minutes=args.minutes)
    print("✅  Demo complete.")