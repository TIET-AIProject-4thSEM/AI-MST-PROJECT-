"""
THE FURNACE: High-Intensity Thermal Simulator
=============================================
Designed to push modern CPUs to 80°C+ using:
- Floating Point Stress (FPU heat)
- Memory Thrashing (RAM/Controller heat)
- Multiprocessing (All cores)

Run predict_realtime.py in another terminal first.
"""

import multiprocessing as mp
import time
import psutil
import os
import sys
import random

class FurnaceWorkload:
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.workers = []
        self.stop_event = mp.Event()

    def _furnace_worker(self, stop_event, intensity):
        """
        Generates maximum heat by mixing FPU and Memory stress.
        """
        # Data block for memory thrashing (1MB of floats)
        # Moving data through caches generates significant heat
        data_block = [random.random() for _ in range(128 * 1024)] 
        
        cycle_time = 0.1  # 100ms chunks
        
        while not stop_event.is_set():
            start_time = time.time()
            
            # If intensity is 100%, we skip the timer check for max throughput
            if intensity >= 0.99:
                # UNLIMITED BURN - No sleep, pure math
                for i in range(1000):
                    # Heavy FPU operation (Power())
                    _ = 1.0001 ** 2.5
                    # Memory access
                    _ = data_block[i % len(data_block)]
            else:
                # DUTY CYCLE BURN
                on_time = cycle_time * intensity
                while (time.time() - start_time) < on_time:
                    # Slightly lighter loop for controlled loads
                    _ = 1.0001 ** 5000 
                
                # Sleep remainder
                elapsed = time.time() - start_time
                sleep_time = cycle_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def set_load(self, percent):
        """Set CPU load with aggressive worker spawning"""
        self.stop()

        if percent <= 0:
            return

        intensity = percent / 100.0
        self.stop_event.clear()

        print(f"  ...Igniting {self.cpu_count} furnace workers...")
        
        for _ in range(self.cpu_count):
            p = mp.Process(target=self._furnace_worker, args=(self.stop_event, intensity))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Give OS time to schedule
        time.sleep(1)
        actual = psutil.cpu_percent(interval=0.5)
        print(f"▶ Load Set: Target {percent}% → Actual {actual:.1f}%")

    def stop(self):
        self.stop_event.set()
        # Aggressive cleanup
        for p in self.workers:
            p.join(timeout=0.2)
            if p.is_alive():
                p.terminate()
        self.workers = []

    def get_cpu_temp(self):
        """Try to get actual CPU temp for display"""
        try:
            temps = psutil.sensors_temperatures()
            # Check common sensor names
            for name in ['coretemp', 'cpu_thermal', 'k10temp', 'zenpower', 'asus']:
                if name in temps:
                    return temps[name][0].current
        except:
            pass
        return 0.0

    def run_demo(self):
        print("""
╔══════════════════════════════════════════════════════╗
║         THE FURNACE - HIGH HEAT SIMULATOR            ║
╚══════════════════════════════════════════════════════╝
Goal: Push CPU > 80°C for realistic critical warnings.

Sequence:
1. Baseline (Idle)
2. Gaming Load (70% - FPU Stress)
3. STRESS TEST (100% - Maximum Heat)
4. Cooldown
""")

        input("Press ENTER to ignite...")
        
        # 1. Idle
        print(f"\n[1/4] IDLE (Warmup) - 30s")
        self.set_load(10)
        self._wait(30)

        # 2. Heavy Gaming
        print(f"\n[2/4] GAMING SIMULATION (75% Load) - 60s")
        self.set_load(75)
        self._wait(60)

        # 3. Maximum Stress
        print(f"\n[3/4] 🔥 MAXIMUM STRESS TEST (100% Load) - 90s")
        print("      (This should hit your 80°C threshold)")
        self.set_load(100)
        self._wait(90)

        # 4. Cooldown
        print(f"\n[4/4] COOLDOWN - 60s")
        self.stop()
        self._wait(60)

        print("\n✅ Simulation Complete.")

    def _wait(self, seconds):
        for i in range(seconds, 0, -1):
            temp = self.get_cpu_temp()
            sys.stdout.write(f"\r  ⏳ {i:2d}s | Temp: {temp:.1f}°C   ")
            sys.stdout.flush()
            time.sleep(1)
        print()

    def cleanup(self):
        self.stop()

if __name__ == "__main__":
    if os.geteuid() != 0 and sys.platform.startswith("linux"):
        print("⚠ NOTE: Run with 'sudo' for better process priority/heating.\n")

    furnace = FurnaceWorkload()
    try:
        furnace.run_demo()
    except KeyboardInterrupt:
        print("\n⚠ Extinguished.")
    finally:
        furnace.cleanup()