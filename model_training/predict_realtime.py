"""
Proactive Thermal Management — Live Demo
=========================================
Run this while your workload generator stresses the CPU.
This script predicts temperature 5 seconds ahead and shows
what the proactive cooling decision would be in real time.

Usage:
    python thermal_demo.py              # runs for 5 minutes
    python thermal_demo.py --minutes 10 # custom duration
    python thermal_demo.py --port COM4  # specify Arduino port

Hardware (optional): REES52 DS18B20 + REES52 L9110 Fan Module
If Arduino is not connected, ambient temperature is simulated.
"""

import psutil
import time
import numpy as np
import pandas as pd
import joblib
import os
import sys
import argparse
import warnings
from datetime import datetime
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── optional Arduino support ─────────────────────────────────────────────────
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = 'models/best_thermal_model.pkl'
SCALER_PATH     = 'models/feature_scaler.pkl'
INFO_PATH       = 'models/model_info.json'
RESULT_CSV      = 'results/demo_log.csv'
RESULT_PLOT     = 'results/demo_summary.png'

TEMP_WARNING    = 70.0   # °C — begin ramping fan
TEMP_CRITICAL   = 80.0   # °C — full fan speed
SAFETY_BUFFER   = 5.0    # °C — added to model RMSE (~3.2°C) as recommended
PREDICT_HORIZON = 5      # seconds ahead


# ═══════════════════════════════════════════════════════════════════════════════
class ThermalDemo:
    """
    Loads a trained Random Forest / tree-ensemble model and runs a 1 Hz
    proactive thermal prediction loop, displaying a colour-coded dashboard
    and saving a summary plot + CSV at the end.
    """

    def __init__(self, arduino_port=None):
        self.model          = None
        self.scaler         = None
        self.feature_names  = None
        self.arduino        = None
        self.arduino_ok     = False
        self.history        = []          # rolling window of state dicts
        self.log            = []          # one entry per prediction tick

        # fan rate-limiting for L9110 H-bridge (max ±20 PWM per second)
        self.last_fan_pwm   = 0
        self.max_fan_step   = 20

        self._load_model()
        if arduino_port:
            self._init_arduino(arduino_port)

        # warm up psutil so first non-blocking call isn't 0%
        psutil.cpu_percent(interval=None)
        time.sleep(0.1)

    # ── model loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"{RED}✗ Model not found at {MODEL_PATH}{RESET}")
            print("  Run your training notebook first, then try again.")
            sys.exit(1)

        self.model  = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        import json
        if os.path.exists(INFO_PATH):
            with open(INFO_PATH) as f:
                info = json.load(f)
            self.feature_names = info['features']
            print(f"{GREEN}✓ Model loaded{RESET}  — {info['model_name']}")
            print(f"  Test RMSE {info['test_rmse']:.3f}°C   |   Test R² {info['test_r2']:.4f}")
        else:
            self.feature_names = None
            print(f"{GREEN}✓ Model loaded{RESET}  (no model_info.json found)")

    # ── Arduino / DS18B20 ─────────────────────────────────────────────────────
    def _init_arduino(self, port):
        if not SERIAL_AVAILABLE:
            print("⚠  pyserial not installed — running without hardware.")
            return

        candidates = [port, '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0',
                      'COM3', 'COM4', 'COM5']
        for p in candidates:
            try:
                self.arduino = serial.Serial(p, 9600, timeout=1)
                time.sleep(2.5)
                self.arduino.reset_input_buffer()
                self.arduino.reset_output_buffer()

                # quick sensor test
                self.arduino.write(b'T\n')
                time.sleep(0.85)
                if self.arduino.in_waiting:
                    raw = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    temp = float(raw)
                    if -55 <= temp <= 125:
                        print(f"{GREEN}✓ DS18B20 on {p}{RESET}  — {temp:.4f}°C")
                        self.arduino_ok = True
                        return
            except Exception:
                continue

        print("⚠  Arduino not found — ambient temperature will be simulated.")

    # ── sensor reads ───────────────────────────────────────────────────────────
    def _cpu_temp(self):
        """Read CPU die temperature from the OS."""
        try:
            sensors = psutil.sensors_temperatures()
            for key in ('coretemp', 'k10temp', 'cpu_thermal'):
                if key in sensors:
                    return sensors[key][0].current
            return list(sensors.values())[0][0].current
        except Exception:
            # software fallback when sensors aren't available (VM / container)
            load = psutil.cpu_percent(interval=None)
            return 35.0 + load * 0.4 + np.random.normal(0, 1.0)

    def _ambient_temp(self):
        """DS18B20 reading or synthetic ambient if no hardware."""
        if self.arduino_ok:
            try:
                self.arduino.reset_input_buffer()
                self.arduino.write(b'T\n')
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1.0:
                    if self.arduino.in_waiting:
                        raw = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                        v = float(raw)
                        if -55 <= v <= 125:
                            return v
                    time.sleep(0.01)
                self.arduino_ok = False   # timed out — fall through
            except Exception:
                self.arduino_ok = False

        # gentle sine-wave simulation (~24°C ± 2°C over the day)
        return 24.0 + 2.0 * np.sin(time.time() / 3600)

    def _system_state(self):
        return {
            'timestamp': time.time(),
            'cpu_temp':  self._cpu_temp(),
            'cpu_load':  psutil.cpu_percent(interval=None),
            'ram_usage': psutil.virtual_memory().percent,
            'ambient':   self._ambient_temp(),
        }

    # ── feature engineering ───────────────────────────────────────────────────
    def _build_features(self, state):
        """Mirror the feature engineering used during training."""
        self.history.append(state)
        if len(self.history) > 30:
            self.history.pop(0)

        # need at least 11 samples for all lag/rolling features
        if len(self.history) < 11:
            return None

        h   = self.history
        f   = {}

        # base
        f['cpu_load']    = state['cpu_load']
        f['ram_usage']   = state['ram_usage']
        f['ambient_temp'] = state['ambient']

        # lag features
        f['cpu_load_lag1']  = h[-2]['cpu_load']
        f['cpu_load_lag5']  = h[-6]['cpu_load']
        f['cpu_load_lag10'] = h[-11]['cpu_load']
        f['cpu_temp_lag1']  = h[-2]['cpu_temp']
        f['cpu_temp_lag5']  = h[-6]['cpu_temp']

        # rate-of-change (thermal momentum)
        f['temp_rate']        = state['cpu_temp'] - h[-2]['cpu_temp']
        f['temp_acceleration'] = f['temp_rate'] - (h[-2]['cpu_temp'] - h[-3]['cpu_temp'])
        f['load_rate']        = state['cpu_load'] - h[-2]['cpu_load']

        # rolling statistics
        loads_10 = [s['cpu_load'] for s in h[-10:]]
        temps_10 = [s['cpu_temp'] for s in h[-10:]]
        loads_all = [s['cpu_load'] for s in h]

        f['cpu_load_roll10'] = np.mean(loads_10)
        f['cpu_temp_roll10'] = np.mean(temps_10)
        f['cpu_load_roll30'] = np.mean(loads_all)
        f['cpu_load_std10']  = np.std(loads_10)

        # physics-informed interactions
        f['load_ambient_interaction'] = state['cpu_load'] * state['ambient']
        f['thermal_stress']           = state['cpu_load'] * state['cpu_temp']
        f['temp_above_ambient']       = state['cpu_temp'] - state['ambient']

        # regime flags
        f['is_high_load'] = 1 if state['cpu_load'] > 70 else 0
        f['is_heating']   = 1 if f['temp_rate'] > 0.5 else 0
        f['is_cooling']   = 1 if f['temp_rate'] < -0.5 else 0

        # cyclic time encoding (prevents hour-number ordering artefact)
        hour = datetime.now().hour
        f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        f['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        return f

    # ── prediction ────────────────────────────────────────────────────────────
    def _predict(self, features):
        df = pd.DataFrame([features])
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                print(f"⚠  Missing features: {missing}")
                return None
            df = df[self.feature_names]
        try:
            return float(self.model.predict(df)[0])
        except Exception:
            try:
                return float(self.model.predict(self.scaler.transform(df))[0])
            except Exception as e:
                print(f"✗ Prediction failed: {e}")
                return None

    # ── fan control ───────────────────────────────────────────────────────────
    def _fan_command(self, predicted_temp, current_temp):
        """Derive fan PWM from predicted temp; send to L9110 if available."""
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

        # smooth transitions — protects L9110 from sudden load spikes
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

    # ── banner ────────────────────────────────────────────────────────────────
    def _print_banner(self, duration_min):
        hw = "DS18B20 + L9110" if self.arduino_ok else "Software simulation"
        print(f"\n{BOLD}{'═'*72}{RESET}")
        print(f"{BOLD}  PROACTIVE THERMAL MANAGEMENT — LIVE DEMO{RESET}")
        print(f"  Hardware  : {hw}")
        print(f"  Duration  : {duration_min} min   |   Sample rate : 1 Hz")
        print(f"  Prediction: {PREDICT_HORIZON}s ahead  |   Safety buffer: ±{SAFETY_BUFFER}°C")
        print(f"  Thresholds: WARNING {TEMP_WARNING}°C  |  CRITICAL {TEMP_CRITICAL}°C")
        print(f"{BOLD}{'═'*72}{RESET}\n")
        print("  Warming up — collecting 11 initial samples…")

    # ── summary plot ──────────────────────────────────────────────────────────
    def _save_summary_plot(self):
        os.makedirs(os.path.dirname(RESULT_PLOT) or '.', exist_ok=True)

        df = pd.DataFrame(self.log)
        df['t'] = range(len(df))

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0d1117')
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3)

        ACTUAL_C  = '#58a6ff'
        PRED_C    = '#f78166'
        LOAD_C    = '#3fb950'
        FAN_C     = '#d2a8ff'

        def _style(ax, title):
            ax.set_facecolor('#161b22')
            ax.set_title(title, color='white', fontsize=11, pad=8)
            ax.tick_params(colors='#8b949e')
            ax.spines[:].set_color('#30363d')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.grid(True, color='#21262d', linewidth=0.8)

        # ── panel 1: temperature trace ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['t'], df['current_temp'],  color=ACTUAL_C, lw=1.4,
                 label='Actual CPU temp', alpha=0.95)
        ax1.plot(df['t'], df['predicted_temp'], color=PRED_C,   lw=1.2,
                 linestyle='--', label=f'Predicted (+{PREDICT_HORIZON}s)', alpha=0.9)
        ax1.fill_between(df['t'],
                         df['predicted_temp'] - SAFETY_BUFFER,
                         df['predicted_temp'] + SAFETY_BUFFER,
                         color=PRED_C, alpha=0.08, label='±5°C safety band')
        ax1.axhline(TEMP_WARNING,  color=YELLOW.strip('\033[m'), ls=':', lw=1,
                    label=f'Warning {TEMP_WARNING}°C',  alpha=0.8)
        ax1.axhline(TEMP_CRITICAL, color=RED.strip('\033[m'), ls=':', lw=1,
                    label=f'Critical {TEMP_CRITICAL}°C', alpha=0.8)
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_xlabel('Sample (seconds)')
        ax1.legend(fontsize=8.5, loc='upper left', facecolor='#1c2128',
                   edgecolor='#30363d', labelcolor='white')
        _style(ax1, 'CPU Temperature — Actual vs Predicted')

        # ── panel 2: CPU load ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df['t'], df['cpu_load'], color=LOAD_C, lw=1.2, alpha=0.9)
        ax2.fill_between(df['t'], 0, df['cpu_load'], color=LOAD_C, alpha=0.18)
        ax2.set_ylabel('CPU Load (%)')
        ax2.set_xlabel('Sample (seconds)')
        ax2.set_ylim(0, 105)
        _style(ax2, 'CPU Utilisation')

        # ── panel 3: fan speed ────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(df['t'], df['fan_pwm'], color=FAN_C, lw=1.2, alpha=0.9)
        ax3.fill_between(df['t'], 0, df['fan_pwm'], color=FAN_C, alpha=0.18)
        ax3.set_ylabel('Fan PWM (0–255)')
        ax3.set_xlabel('Sample (seconds)')
        ax3.set_ylim(0, 275)
        _style(ax3, 'L9110 Fan Speed (Proactive Response)')

        # summary text box
        d = df['predicted_delta']
        info = (
            f"Samples : {len(df)}\n"
            f"Temp range : {df['current_temp'].min():.1f}°C – {df['current_temp'].max():.1f}°C\n"
            f"Mean |Δ| : {d.abs().mean():.2f}°C\n"
            f"Max  |Δ| : {d.abs().max():.2f}°C\n"
            f"Ambient : {df['ambient_temp'].mean():.1f}°C avg\n"
            f"Fan range : {df['fan_pwm'].min()}–{df['fan_pwm'].max()} PWM"
        )
        fig.text(0.77, 0.10, info, fontsize=8.5, color='#c9d1d9',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#1c2128',
                           edgecolor='#30363d'))

        fig.suptitle('Proactive Thermal Management — Demo Summary',
                     color='white', fontsize=14, fontweight='bold', y=0.98)

        plt.savefig(RESULT_PLOT, dpi=140, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\n{GREEN}✓ Summary plot saved → {RESULT_PLOT}{RESET}")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self, duration_minutes=5):
        self._print_banner(duration_minutes)

        os.makedirs(os.path.dirname(RESULT_CSV) or '.', exist_ok=True)

        end_time        = time.monotonic() + duration_minutes * 60
        next_tick       = time.monotonic()
        sample_n        = 0
        header_printed  = False

        try:
            while time.monotonic() < end_time:
                state    = self._system_state()
                sample_n += 1
                features = self._build_features(state)

                # ── warm-up phase ──────────────────────────────────────────────
                if features is None:
                    n = len(self.history)
                    bar = '█' * n + '░' * (11 - n)
                    print(f"\r  [{bar}] {n}/11", end='', flush=True)
                    next_tick += 1.0
                    _sleep_until(next_tick)
                    continue

                # ── first prediction — print header ───────────────────────────
                if not header_printed:
                    print(f"\n\n{'─'*72}")
                    print(f"  {'Time':8s}  {'Current':>9s}  {'Predicted':>11s}  "
                          f"{'Δ (5s)':>8s}  {'Status':10s}  {'Fan PWM':>8s}  {'Load':>6s}")
                    print(f"{'─'*72}")
                    header_printed = True

                predicted = self._predict(features)
                if predicted is None:
                    next_tick += 1.0
                    _sleep_until(next_tick)
                    continue

                delta = predicted - state['cpu_temp']
                pwm, label, colour = self._fan_command(predicted, state['cpu_temp'])
                ts    = datetime.now().strftime('%H:%M:%S')

                print(
                    f"  {ts}  "
                    f"{state['cpu_temp']:7.2f}°C  "
                    f"{predicted:9.2f}°C  "
                    f"{delta:+7.2f}°C  "
                    f"{colour}{label}{RESET}  "
                    f"{pwm:5d}/255  "
                    f"{state['cpu_load']:4.0f}%",
                    flush=True
                )

                self.log.append({
                    'timestamp':      ts,
                    'current_temp':   state['cpu_temp'],
                    'predicted_temp': predicted,
                    'predicted_delta': delta,
                    'cpu_load':       state['cpu_load'],
                    'ambient_temp':   state['ambient'],
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

    # ── cleanup + final report ────────────────────────────────────────────────
    def _shutdown(self):
        # safe fan stop
        if self.arduino_ok and self.arduino:
            try:
                self.arduino.write(b'F0\n')
                time.sleep(0.1)
                self.arduino.close()
            except Exception:
                pass

        if not self.log:
            print("No data collected.")
            return

        df = pd.DataFrame(self.log)
        df.to_csv(RESULT_CSV, index=False)

        d = df['predicted_delta'].abs()
        warnings_n  = (df['status'] == 'WARNING').sum()
        critical_n  = (df['status'] == 'CRITICAL').sum()

        print(f"\n{BOLD}{'═'*72}{RESET}")
        print(f"{BOLD}  DEMO SUMMARY{RESET}")
        print(f"{'─'*72}")
        print(f"  {'Total predictions':30s}: {len(df)}")
        print(f"  {'Temperature range':30s}: {df['current_temp'].min():.1f}°C – "
              f"{df['current_temp'].max():.1f}°C")
        print(f"  {'Mean predicted delta':30s}: {d.mean():.2f}°C")
        print(f"  {'Max  predicted delta':30s}: {d.max():.2f}°C")
        print(f"  {'Ambient (DS18B20/sim)':30s}: {df['ambient_temp'].mean():.2f}°C avg")
        print(f"  {'Fan PWM range':30s}: {df['fan_pwm'].min()} – {df['fan_pwm'].max()} / 255")
        print(f"  {'WARNING-level ticks':30s}: {warnings_n}")
        print(f"  {'CRITICAL-level ticks':30s}: {critical_n}")
        print(f"{'─'*72}")
        print(f"  {GREEN}✓ Log saved → {RESULT_CSV}{RESET}")

        self._save_summary_plot()
        print(f"{BOLD}{'═'*72}{RESET}\n")


# ── helpers ───────────────────────────────────────────────────────────────────
def _sleep_until(target):
    """Sleep until a monotonic timestamp, handling any overshoot gracefully."""
    remaining = target - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║        PROACTIVE THERMAL MANAGEMENT  ·  LIVE DEMO           ║
║  Physics-aware ML predicts CPU temperature {PREDICT_HORIZON}s ahead         ║
║  and sets fan speed BEFORE thresholds are hit.              ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

    parser = argparse.ArgumentParser(description='Thermal prediction live demo')
    parser.add_argument('--minutes', type=int,  default=5,
                        help='Monitoring duration in minutes (default: 5)')
    parser.add_argument('--port',    type=str,  default=None,
                        help='Arduino serial port (e.g. COM4 or /dev/ttyUSB0)')
    args = parser.parse_args()

    demo = ThermalDemo(arduino_port=args.port)
    print(f"\n{BOLD}▶ Starting {args.minutes}-minute demo…{RESET}")
    print("  Launch your workload generator now, then watch the predictions.")
    print("  Press Ctrl+C at any time to stop early and see the summary.\n")
    time.sleep(2)

    demo.run(duration_minutes=args.minutes)
    print("✅  Demo complete.")