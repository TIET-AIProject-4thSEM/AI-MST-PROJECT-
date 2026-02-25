"""
Proactive Thermal Management — Live Demo
=========================================
Run this while your workload generator stresses the CPU.

HOW THE PROACTIVE COOLING WORKS
─────────────────────────────────
The model predicts the *current* CPU temperature from load signals
(utilisation + its lag history + smoothed ambient).

It does not output a static value like "82°C". It outputs something
semantically richer: "82°C and rising" or "82°C and falling".

This comes from the lag features. If cpu_util_lag1, lag3, and lag6 are
all high and climbing, the model outputs a HIGHER estimate than if the
same current utilisation were preceded by a falling trajectory — even
though the instantaneous reading is identical. That asymmetry encodes
thermal inertia.

Fan control uses BOTH outputs:

  Estimate  →  sets base PWM (where is the temp right now?)
  Δ = estimate − actual  →  trend boost/cut (which direction is it heading?)

  Δ > 0  →  heating trajectory  →  boost fan above base
  Δ ≈ 0  →  stable              →  hold base
  Δ < 0  →  cooling trajectory  →  reduce fan below base

This means fan response is driven by load trend, not by the temperature
reading crossing a threshold — which is the difference between proactive
and reactive control.

Usage:
    python thermal_demo.py               # 5-minute run, auto-detects Arduino
    python thermal_demo.py --minutes 10
    python thermal_demo.py --port COM4   # hint a specific port (still auto-scans)

Hardware (optional): REES52 DS18B20 + REES52 L9110 fan module.
If no Arduino is found the script warns clearly and falls back to a
synthetic ambient signal — all other functionality is unchanged.
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

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── ANSI ─────────────────────────────────────────────────────────────────────
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
TEMP_WARNING  = 70.0
TEMP_CRITICAL = 80.0
SAFETY_BUFFER = 5.0    # °C safety band on plot (~model RMSE × 1.5)

# trend boost: each 1°C of delta adds/removes this many PWM units
TREND_GAIN      =  3.0
TREND_REDUCTION =  2.0
MAX_TREND_BOOST =  40
MAX_TREND_CUT   =  25

# L9110 rate limiting
MAX_FAN_STEP = 20
MIN_FAN_PWM  = 30

WARMUP_SAMPLES = 8   # lag-6 needs index -7, plus one guard sample


# ═════════════════════════════════════════════════════════════════════════════
class ThermalDemo:

    def __init__(self, arduino_port=None):
        self.model         = None
        self.scaler        = None
        self.feature_names = None
        self.model_rmse    = 3.24

        self.arduino    = None
        self.arduino_ok = False

        self.history      = deque(maxlen=30)
        self.log          = []
        self.last_fan_pwm = MIN_FAN_PWM
        self.max_fan_step = MAX_FAN_STEP

        self._load_model()
        self._init_arduino(arduino_port)   # always scans; --port just goes first

        psutil.cpu_percent(interval=None)  # prime non-blocking call
        time.sleep(0.1)

    # ── model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        for path in (MODEL_PATH, SCALER_PATH):
            if not os.path.exists(path):
                print(f"{RED}✗ Not found: {path}{RESET}")
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
            print(f"  Test RMSE : {info['test_rmse']:.3f}°C  |  R² : {info['test_r2']:.4f}")
            print(f"  Features  : {self.feature_names}")
        else:
            print(f"{YELLOW}⚠  model_info.json missing — feature order inferred.{RESET}")

<<<<<<< HEAD
    # ── Arduino auto-detection ────────────────────────────────────────────────
    def _init_arduino(self, preferred_port=None):
        """
        Auto-detect Arduino + DS18B20 on Fedora (ttyUSB/ttyACM) and Windows (COM).

        Port scan order:
          1. preferred_port from --port arg, if given
          2. USB serial devices found by serial.tools.list_ports
             (filters for CH340, CP210x, FTDI, ACM — all common Arduino chips)
          3. Fixed fallback list for machines where list_ports misses the device

        Every tried port prints a one-line result so the user always knows
        what was attempted. Silent failures are only suppressed for fallback
        ports that simply don't exist on this machine (SerialException).
        """
        if not SERIAL_AVAILABLE:
            print(f"\n{YELLOW}⚠  pyserial not installed — no hardware control.{RESET}")
            print("    pip install pyserial")
            print("    Falling back to synthetic ambient temperature.\n")
            return

        print(f"\n{BOLD}── Arduino auto-detection ──────────────────────────────────{RESET}")

        candidates = []

        if preferred_port:
            candidates.append(preferred_port)
            print(f"  Priority port (--port arg): {preferred_port}")

        # ask the OS which USB serial devices are plugged in right now
        try:
            usb_ports = []
            for p in serial.tools.list_ports.comports():
                desc = (p.description or '').lower()
                hwid = (p.hwid or '').lower()
                is_arduino = any(kw in desc or kw in hwid for kw in
                                 ('arduino', 'ch340', 'ch341', 'cp210',
                                  'ftdi', 'usb serial', 'acm'))
                if is_arduino or 'usb' in hwid:
                    usb_ports.append(p.device)

            if usb_ports:
                print(f"  USB serial devices detected: {usb_ports}")
            else:
                print("  No USB serial devices detected — trying fallback list.")

            for p in usb_ports:
                if p not in candidates:
                    candidates.append(p)
        except Exception as e:
            print(f"  {YELLOW}list_ports scan failed ({e}) — using fallback list.{RESET}")

        # fixed fallback: Fedora/Linux first, then Windows
        for p in ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
                  '/dev/ttyACM0', '/dev/ttyACM1',
                  'COM3', 'COM4', 'COM5', 'COM6', 'COM7']:
            if p not in candidates:
                candidates.append(p)

        print(f"  Trying {len(candidates)} port(s)…\n")

        for port in candidates:
            try:
                conn = serial.Serial(port, 9600, timeout=1)
                time.sleep(2.5)          # wait for Arduino bootloader
                conn.reset_input_buffer()
                conn.reset_output_buffer()

                conn.write(b'T\n')
                time.sleep(0.9)          # DS18B20 needs ~750 ms at 12-bit

                if conn.in_waiting:
                    raw  = conn.readline().decode('utf-8', errors='ignore').strip()
=======
    # ── Arduino ───────────────────────────────────────────────────────────────
    def _init_arduino(self, port):
        if not SERIAL_AVAILABLE:
            print("⚠  pyserial not installed — no hardware control.")
            return
        for p in [port, '/dev/ttyUSB0', '/dev/ttyUSB1',
                  '/dev/ttyACM0', 'COM3', 'COM4', 'COM5']:
            try:
                self.arduino = serial.Serial(p, 9600, timeout=1)
                time.sleep(2.5)
                self.arduino.reset_input_buffer()
                self.arduino.reset_output_buffer()
                self.arduino.write(b'T\n')
                time.sleep(0.85)
                if self.arduino.in_waiting:
                    raw  = self.arduino.readline().decode('utf-8', errors='ignore').strip()
>>>>>>> parent of 00be971 (the results seem worse than previous predict realtime)
                    temp = float(raw)
                    if -55 <= temp <= 125:
                        print(f"{GREEN}✓ DS18B20 on {p}{RESET}  — {temp:.4f}°C")
                        self.arduino_ok = True
                        return
<<<<<<< HEAD
                    else:
                        print(f"  {YELLOW}✗ {port:<18}{RESET} — value out of DS18B20 range: {raw!r}")
                        conn.close()
                else:
                    print(f"  {YELLOW}✗ {port:<18}{RESET} — no response from DS18B20")
                    conn.close()

            except serial.SerialException:
                # port doesn't exist or is busy — only report if user specified it
                if preferred_port and port == preferred_port:
                    print(f"  {RED}✗ {port:<18}{RESET} — could not open port")
            except ValueError:
                print(f"  {YELLOW}✗ {port:<18}{RESET} — response not a valid float: {raw!r}")
            except Exception as e:
                print(f"  {YELLOW}✗ {port:<18}{RESET} — {e}")

        # nothing worked
        print(f"\n  {YELLOW}⚠  No Arduino with DS18B20 found on any port.{RESET}")
        print("     Common fixes on Fedora:")
        print("       sudo usermod -aG dialout $USER   (then log out and back in)")
        print("       sudo chmod a+rw /dev/ttyUSB0")
        print("     Check the Arduino sketch responds to 'T\\n' at 9600 baud.")
        print(f"\n  Falling back to synthetic ambient (24°C ± 2°C sine wave).")
        print(f"{'─'*60}\n")
=======
            except Exception:
                continue
        print("⚠  Arduino not found — ambient simulated, fan commands logged only.")
>>>>>>> parent of 00be971 (the results seem worse than previous predict realtime)

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
        Exactly mirrors the notebook's feature engineering:

            cpu_utilization_smooth  — 5-sample rolling mean
            memory_usage_smooth     — 5-sample rolling mean
            ambient_temp_smooth     — 5-sample rolling mean
            hour                    — raw integer 0-23
            day_of_week             — Mon=0 … Sun=6
            is_business             — 1 if hour 8-18 on a weekday
            cpu_util_lag1           — utilisation 1 s ago
            cpu_util_lag3           — utilisation 3 s ago
            cpu_util_lag6           — utilisation 6 s ago

        The lag features give the model its trend awareness:
        rising lags → higher estimate than a flat or falling trajectory
        at the same instantaneous utilisation value.
        """
        self.history.append(snap)
        n = len(self.history)
        if n < WARMUP_SAMPLES:
            return None

        h = list(self.history)   # h[-1] = newest

        w              = min(5, n)
        cpu_smooth     = np.mean([s['cpu_util'] for s in h[-w:]])
        memory_smooth  = np.mean([s['memory']   for s in h[-w:]])
        ambient_smooth = np.mean([s['ambient']  for s in h[-w:]])

        lag1 = h[-2]['cpu_util']   # 1 s ago
        lag3 = h[-4]['cpu_util']   # 3 s ago
        lag6 = h[-7]['cpu_util']   # 6 s ago

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
                return None
            df = df[self.feature_names]
        try:
            return float(self.model.predict(df)[0])
        except Exception:
            try:
                return float(self.model.predict(self.scaler.transform(df))[0])
            except Exception as e:
                print(f"{RED}✗ Prediction error: {e}{RESET}")
                return None

    # ── fan control ───────────────────────────────────────────────────────────
    def _fan_command(self, estimate, actual):
        """
        Two-signal fan control:

        Signal 1 — ESTIMATE sets the base PWM zone:
            < 60°C      →  base 50   (quiet)
            60–70°C     →  base 100  (elevated)
            70–80°C     →  base 128–254 (warning ramp)
            ≥ 80°C      →  base 255  (maximum)

        Signal 2 — DELTA (estimate − actual) applies a trend correction:
            Δ > 0  →  boost  by Δ × TREND_GAIN      (cap +40 PWM)
            Δ < 0  →  reduce by |Δ| × TREND_REDUCTION (cap −25 PWM)
            Δ ≈ 0  →  no correction

        Combined PWM is rate-limited (±20/s) and floored at MIN_FAN_PWM.
        """
        delta = estimate - actual

        # signal 1: base from estimate
        if estimate >= TEMP_CRITICAL:
            base, zone, colour = 255, "CRITICAL", RED
        elif estimate >= TEMP_WARNING:
            ratio = (estimate - TEMP_WARNING) / (TEMP_CRITICAL - TEMP_WARNING)
            base  = int(128 + 127 * ratio)
            zone, colour = "WARNING", YELLOW
        elif estimate >= 60:
            base, zone, colour = 100, "ELEVATED", BLUE
        else:
            base, zone, colour = 50, "NORMAL", GREEN

        # signal 2: trend correction from delta
        if delta > 0.5:
            trend_pwm = int(min(delta * TREND_GAIN, MAX_TREND_BOOST))
            trend_sym = "↑"
        elif delta < -0.5:
            trend_pwm = -int(min(abs(delta) * TREND_REDUCTION, MAX_TREND_CUT))
            trend_sym = "↓"
        else:
            trend_pwm = 0
            trend_sym = "→"

        # combine, rate-limit, floor
        target = int(np.clip(base + trend_pwm, MIN_FAN_PWM, 255))
        pwm    = int(np.clip(target,
                             self.last_fan_pwm - self.max_fan_step,
                             self.last_fan_pwm + self.max_fan_step))
        self.last_fan_pwm = pwm

        if self.arduino_ok:
            try:
                self.arduino.reset_output_buffer()
                self.arduino.write(f'F{pwm}\n'.encode())
            except Exception:
                self.arduino_ok = False

        return pwm, base, trend_pwm, f"{zone} {trend_sym}", colour

    # ── summary plot ──────────────────────────────────────────────────────────
    def _save_plot(self):
        os.makedirs(os.path.dirname(RESULT_PLOT) or '.', exist_ok=True)
        df = pd.DataFrame(self.log)
        df['t'] = range(len(df))

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor('#0d1117')
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                hspace=0.48, wspace=0.32,
                                height_ratios=[2, 1, 1])

        def _style(ax, title):
            ax.set_facecolor('#161b22')
            ax.set_title(title, color='white', fontsize=10, pad=7)
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.spines[:].set_color('#30363d')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.grid(True, color='#21262d', linewidth=0.7)

        # panel 1: temperature trace — full width
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['t'], df['actual_temp'], color='#58a6ff', lw=1.5,
                 label='Actual CPU temp', alpha=0.95)
        ax1.plot(df['t'], df['estimate'],    color='#f78166', lw=1.2,
                 ls='--', label='Model estimate (trend-aware)', alpha=0.9)
        ax1.fill_between(df['t'],
                         df['estimate'] - SAFETY_BUFFER,
                         df['estimate'] + SAFETY_BUFFER,
                         color='#f78166', alpha=0.07,
                         label=f'±{SAFETY_BUFFER}°C safety band')
        ax1.axhline(TEMP_WARNING,  color='#e3b341', ls=':', lw=1.2, alpha=0.8,
                    label=f'Warning {TEMP_WARNING}°C')
        ax1.axhline(TEMP_CRITICAL, color='#f85149', ls=':', lw=1.2, alpha=0.8,
                    label=f'Critical {TEMP_CRITICAL}°C')
        ax1.set_ylabel('Temperature (°C)', fontsize=9)
        ax1.set_xlabel('Sample (seconds)', fontsize=9)
        ax1.legend(fontsize=8, loc='upper left',
                   facecolor='#1c2128', edgecolor='#30363d', labelcolor='white')
        _style(ax1, 'CPU Temperature — Actual vs Trend-Aware Model Estimate')

        # panel 2: delta (trend signal)
        ax_d = fig.add_subplot(gs[1, 0])
        pos = df['delta'].clip(lower=0)
        neg = df['delta'].clip(upper=0)
        ax_d.fill_between(df['t'], 0, pos, color='#f78166', alpha=0.7,
                          label='Heating (Δ > 0)')
        ax_d.fill_between(df['t'], 0, neg, color='#58a6ff', alpha=0.7,
                          label='Cooling (Δ < 0)')
        ax_d.plot(df['t'], df['delta'], color='#e6edf3', lw=0.8, alpha=0.6)
        ax_d.axhline(0, color='#8b949e', lw=0.8, ls='--')
        ax_d.set_ylabel('Δ Estimate − Actual (°C)', fontsize=8)
        ax_d.set_xlabel('Sample (seconds)', fontsize=8)
        ax_d.legend(fontsize=7.5, facecolor='#1c2128',
                    edgecolor='#30363d', labelcolor='white')
        _style(ax_d, 'Trend Signal  (proactive trigger)')

        # panel 3: CPU load
        ax_l = fig.add_subplot(gs[1, 1])
        ax_l.plot(df['t'], df['cpu_load'], color='#3fb950', lw=1.2, alpha=0.9)
        ax_l.fill_between(df['t'], 0, df['cpu_load'], color='#3fb950', alpha=0.18)
        ax_l.set_ylim(0, 105)
        ax_l.set_ylabel('CPU Utilisation (%)', fontsize=8)
        ax_l.set_xlabel('Sample (seconds)', fontsize=8)
        _style(ax_l, 'CPU Utilisation')

        # panel 4: fan PWM breakdown
        ax_f = fig.add_subplot(gs[2, :])
        ax_f.plot(df['t'], df['fan_pwm'],  color='#d2a8ff', lw=1.5,
                  label='Final PWM (rate-limited)', alpha=0.95)
        ax_f.plot(df['t'], df['base_pwm'], color='#79c0ff', lw=1.0,
                  ls='--', label='Base PWM (from estimate)', alpha=0.7)
        ax_f.fill_between(df['t'], df['base_pwm'], df['fan_pwm'],
                          where=(df['fan_pwm'] > df['base_pwm']),
                          color='#f78166', alpha=0.25, label='Trend boost (+)')
        ax_f.fill_between(df['t'], df['fan_pwm'], df['base_pwm'],
                          where=(df['fan_pwm'] < df['base_pwm']),
                          color='#58a6ff', alpha=0.25, label='Trend reduction (−)')
        ax_f.set_ylim(0, 275)
        ax_f.set_ylabel('Fan PWM (0–255)', fontsize=9)
        ax_f.set_xlabel('Sample (seconds)', fontsize=9)
        ax_f.legend(fontsize=8, loc='upper left',
                    facecolor='#1c2128', edgecolor='#30363d', labelcolor='white')
        _style(ax_f, 'L9110 Fan Control — Base PWM + Trend Correction')

        delta = df['delta']
        stats = (
            f"Samples      : {len(df)}\n"
            f"Actual range : {df['actual_temp'].min():.1f}–{df['actual_temp'].max():.1f}°C\n"
            f"Mean Δ       : {delta.mean():+.2f}°C\n"
            f"Max  |Δ|     : {delta.abs().max():.2f}°C\n"
            f"Heating ticks: {(delta > 0.5).sum()}\n"
            f"Cooling ticks: {(delta < -0.5).sum()}\n"
            f"Fan range    : {df['fan_pwm'].min()}–{df['fan_pwm'].max()} PWM"
        )
        fig.text(0.80, 0.05, stats, fontsize=8, color='#c9d1d9',
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='#1c2128', edgecolor='#30363d'))
        fig.suptitle('Proactive Thermal Management — Demo Summary',
                     color='white', fontsize=13, fontweight='bold', y=0.99)

        plt.savefig(RESULT_PLOT, dpi=140, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"{GREEN}✓ Plot saved → {RESULT_PLOT}{RESET}")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self, duration_minutes=5):
        hw = "DS18B20 + L9110" if self.arduino_ok else "Software simulation"
        print(f"\n{BOLD}{'═'*76}{RESET}")
        print(f"{BOLD}  PROACTIVE THERMAL MANAGEMENT — LIVE DEMO{RESET}")
        print(f"  Hardware   : {hw}")
        print(f"  Duration   : {duration_minutes} min  |  1 Hz sample rate")
        print(f"  Thresholds : WARNING {TEMP_WARNING}°C  |  CRITICAL {TEMP_CRITICAL}°C")
        print(f"  Model RMSE : ~{self.model_rmse:.2f}°C  |  Safety band: ±{SAFETY_BUFFER}°C")
        print(f"\n  FAN LOGIC  : Base PWM set by estimate.  Trend Δ adds a boost (↑)")
        print(f"               or reduction (↓) on top.  Rate-limited ±{MAX_FAN_STEP} PWM/s.")
        print(f"{BOLD}{'═'*76}{RESET}\n")
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
                    print(f"\n\n{'─'*84}")
                    print(
                        f"  {'Time':8s}  "
                        f"{'Actual':>7s}  "
                        f"{'Estimate':>9s}  "
                        f"{'Δ(est-act)':>10s}  "
                        f"{'Trend':>5s}  "
                        f"{'Status':12s}  "
                        f"{'Base':>4s}  "
                        f"{'PWM':>7s}  "
                        f"{'Load':>5s}"
                    )
                    print(f"{'─'*84}")
                    header_printed = True

                estimate = self._predict(features)
                if estimate is None:
                    next_tick += 1.0
                    _sleep_until(next_tick)
                    continue

                actual = snap['cpu_temp']
                delta  = estimate - actual
                pwm, base_pwm, trend_pwm, label, colour = self._fan_command(estimate, actual)

                if delta > 0.5:
                    trend_disp = f"{RED}↑{RESET} {delta:+.1f}°C"
                elif delta < -0.5:
                    trend_disp = f"{BLUE}↓{RESET} {delta:+.1f}°C"
                else:
                    trend_disp = f"{GREEN}→{RESET} {delta:+.1f}°C"

                ts = datetime.now().strftime('%H:%M:%S')

                print(
                    f"  {ts}  "
                    f"{actual:5.1f}°C  "
                    f"{estimate:7.1f}°C  "
                    f"{delta:+8.2f}°C  "
                    f"  {trend_disp}  "
                    f"{colour}{label:<12}{RESET}  "
                    f"{base_pwm:4d}  "
                    f"{pwm:4d}/255  "
                    f"{snap['cpu_util']:4.0f}%",
                    flush=True
                )

                self.log.append({
                    'timestamp':   ts,
                    'actual_temp': actual,
                    'estimate':    estimate,
                    'delta':       delta,
                    'base_pwm':    base_pwm,
                    'trend_pwm':   trend_pwm,
                    'fan_pwm':     pwm,
                    'status':      label,
                    'cpu_load':    snap['cpu_util'],
                    'ambient_temp': snap['ambient'],
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
            print("No predictions recorded.")
            return

        df = pd.DataFrame(self.log)
        df.to_csv(RESULT_CSV, index=False)

        delta     = df['delta']
        n_heating = (delta > 0.5).sum()
        n_cooling = (delta < -0.5).sum()
        n_stable  = len(df) - n_heating - n_cooling

        print(f"\n{BOLD}{'═'*76}{RESET}")
        print(f"{BOLD}  DEMO SUMMARY{RESET}")
        print(f"{'─'*76}")
        print(f"  {'Total predictions':34s}: {len(df)}")
        print(f"  {'Actual temp range':34s}: "
              f"{df['actual_temp'].min():.1f}°C – {df['actual_temp'].max():.1f}°C")
        print(f"  {'Estimate range':34s}: "
              f"{df['estimate'].min():.1f}°C – {df['estimate'].max():.1f}°C")
        print(f"  {'Mean Δ (estimate − actual)':34s}: {delta.mean():+.2f}°C")
        print(f"  {'Max  |Δ|':34s}: {delta.abs().max():.2f}°C")
        print(f"  {'Heating ticks  (Δ > +0.5°C)':34s}: {n_heating}  {RED}↑{RESET}")
        print(f"  {'Stable  ticks  (|Δ| ≤ 0.5°C)':34s}: {n_stable}  {GREEN}→{RESET}")
        print(f"  {'Cooling ticks  (Δ < −0.5°C)':34s}: {n_cooling}  {BLUE}↓{RESET}")
        print(f"  {'Fan PWM range':34s}: "
              f"{df['fan_pwm'].min()} – {df['fan_pwm'].max()} / 255")
        print(f"  {'Ambient avg (DS18B20/sim)':34s}: {df['ambient_temp'].mean():.2f}°C")
        print(f"{'─'*76}")
        print(f"  {GREEN}✓ Log saved → {RESULT_CSV}{RESET}")

        self._save_plot()
        print(f"{BOLD}{'═'*76}{RESET}\n")


# ── helpers ───────────────────────────────────────────────────────────────────
def _sleep_until(t):
    remaining = t - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"""
\033[1m\033[96m╔══════════════════════════════════════════════════════════════════╗
║       PROACTIVE THERMAL MANAGEMENT  ·  LIVE DEMO              ║
║                                                                ║
║  Estimate  →  base fan PWM  (where is temp now?)              ║
║  Δ = estimate − actual  →  trend boost / cut  (↑ ↓ →)        ║
║                                                                ║
║  Rising lags in model → higher estimate → fan boosts early    ║
╚══════════════════════════════════════════════════════════════════╝\033[0m
""")

    parser = argparse.ArgumentParser(description='Thermal prediction live demo')
    parser.add_argument('--minutes', type=int, default=5,
                        help='Duration in minutes (default: 5)')
    parser.add_argument('--port', type=str, default=None,
                        help='Arduino port hint, e.g. COM4 or /dev/ttyUSB0')
    args = parser.parse_args()

    demo = ThermalDemo(arduino_port=args.port)
    print(f"\n\033[1m▶ Starting {args.minutes}-minute demo…\033[0m")
    print("  Launch your workload generator now.")
    print("  Watch the Trend column — fan responds to ↑ before temp peaks.\n")
    time.sleep(2)

    demo.run(duration_minutes=args.minutes)
    print("✅  Demo complete.")