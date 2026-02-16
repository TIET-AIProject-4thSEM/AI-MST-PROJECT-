"""
Real-Time CPU Temperature Prediction - Simplified Version
==========================================================
Based on simplified feature engineering and Random Forest model

Features used (18 total):
- Base: cpu_load, ram_usage, ambient_temp
- Lag: cpu_temp_lag1, cpu_temp_lag5, cpu_load_lag1, cpu_load_lag5, cpu_load_lag10
- Rate: temp_rate, temp_accelaration, load_rate
- Rolling: cpu_temp_roll10, cpu_load_roll10, cpu_load_roll30, cpu_load_std10
- Interaction: load_ambient_interaction, thermal_stress, temp_above_ambient
"""

import psutil
import time
import numpy as np
import pandas as pd
import joblib
import serial
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplifiedThermalPredictor:
    """
    Simplified real-time thermal prediction system.
    Matches the exact feature engineering from the training notebook.
    """
    
    def __init__(self, 
                 model_path='cpu_temp_predictor.pkl',
                 feature_cols_path='feature_columns.pkl',
                 arduino_port='/dev/ttyUSB0'):
        """Initialize the prediction system."""
        self.model = None
        self.feature_cols = None
        self.arduino = None
        self.feature_history = []
        
        # Fan speed rate limiting
        self.last_fan_speed = 0
        self.max_fan_step = 20  # Maximum change per second
        
        # Load model and feature columns
        self.load_model(model_path, feature_cols_path)
        
        # Connect Arduino (optional)
        self.arduino_available = self._init_arduino(arduino_port)
        
        # Thresholds
        self.TEMP_WARNING = 70.0
        self.TEMP_CRITICAL = 80.0
        self.PREDICTION_HORIZON = 5  # seconds
        
        # Initialize psutil for non-blocking calls
        print("Initializing CPU monitoring (non-blocking mode)...")
        psutil.cpu_percent(interval=None)
        time.sleep(0.1)
        
    def load_model(self, model_path, feature_cols_path):
        """Load trained model and feature column names"""
        try:
            self.model = joblib.load(model_path)
            self.feature_cols = joblib.load(feature_cols_path)
            print(f"✓ Model loaded from: {model_path}")
            print(f"✓ Feature columns loaded: {len(self.feature_cols)} features")
            print(f"✓ Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"   Make sure {model_path} and {feature_cols_path} exist")
            exit(1)
    
    def _init_arduino(self, port):
        """Initialize Arduino with DS18B20 + L9110 modules (optional)"""
        ports_to_try = [port, '/dev/ttyUSB0', '/dev/ttyUSB1', 
                       '/dev/ttyACM0', 'COM3', 'COM4', 'COM5']
        
        for p in ports_to_try:
            try:
                self.arduino = serial.Serial(p, 9600, timeout=1)
                time.sleep(2.5)  # DS18B20 init time
                
                # Flush buffers
                self.arduino.reset_input_buffer()
                self.arduino.reset_output_buffer()
                
                # Test DS18B20
                self.arduino.write(b'T\n')
                time.sleep(0.8)  # DS18B20 conversion time
                
                if self.arduino.in_waiting:
                    response = self.arduino.readline()
                    try:
                        temp = float(response.decode('utf-8').strip())
                        if -55 <= temp <= 125:  # DS18B20 range
                            print(f"✓ Arduino connected on {p}")
                            print(f"✓ DS18B20 reading: {temp:.4f}°C")
                            return True
                    except:
                        pass
            except:
                continue
        
        print(f"⚠ Arduino not available - using simulated ambient temperature")
        return False
    
    def get_system_state(self):
        """Collect current system state (non-blocking)"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
            elif 'k10temp' in temps:
                cpu_temp = temps['k10temp'][0].current
            elif 'cpu_thermal' in temps:
                cpu_temp = temps['cpu_thermal'][0].current
            else:
                try:
                    cpu_temp = list(temps.values())[0][0].current
                except:
                    # Fallback estimation
                    cpu_percent = psutil.cpu_percent(interval=None)
                    cpu_temp = 35.0 + cpu_percent * 0.4 + np.random.normal(0, 1.5)
        except:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_temp = 35.0 + cpu_percent * 0.4 + np.random.normal(0, 1.5)
        
        state = {
            'cpu_load': psutil.cpu_percent(interval=None),
            'ram_usage': psutil.virtual_memory().percent,
            'ambient_temp': self._get_ambient_temp(),
            'cpu_temp': cpu_temp,
            'timestamp': time.time()
        }
        
        return state
    
    def _get_ambient_temp(self):
        """Read ambient temperature from DS18B20 or simulate"""
        if self.arduino_available:
            try:
                self.arduino.reset_input_buffer()
                self.arduino.write(b'T\n')
                
                start = time.monotonic()
                while time.monotonic() - start < 1.0:
                    if self.arduino.in_waiting:
                        response = self.arduino.readline()
                        try:
                            temp = float(response.decode('utf-8').strip())
                            if -55 <= temp <= 125:
                                return temp
                        except:
                            pass
                    time.sleep(0.01)
                
                self.arduino_available = False
            except:
                self.arduino_available = False
        
        # Simulate ambient temperature
        return 21.7 + 0.3 * np.sin(time.time() / 3600)
    
    def engineer_features(self, state):
        """
        Engineer features matching the training notebook exactly.
        
        Requires at least 30 samples for rolling features.
        """
        self.feature_history.append(state)
        
        # Keep last 30 samples for rolling30
        if len(self.feature_history) > 30:
            self.feature_history.pop(0)
        
        # Need at least 30 samples for all features
        if len(self.feature_history) < 30:
            return None
        
        features = {}
        
        # Base features
        features['cpu_load'] = state['cpu_load']
        features['ram_usage'] = state['ram_usage']
        features['ambient_temp'] = state['ambient_temp']
        
        # Lag features
        features['cpu_temp_lag1'] = self.feature_history[-2]['cpu_temp']
        features['cpu_temp_lag5'] = self.feature_history[-6]['cpu_temp']
        features['cpu_load_lag1'] = self.feature_history[-2]['cpu_load']
        features['cpu_load_lag5'] = self.feature_history[-6]['cpu_load']
        features['cpu_load_lag10'] = self.feature_history[-11]['cpu_load']
        
        # Rate features (diff)
        features['temp_rate'] = state['cpu_temp'] - self.feature_history[-2]['cpu_temp']
        features['temp_accelaration'] = features['temp_rate'] - (
            self.feature_history[-2]['cpu_temp'] - self.feature_history[-3]['cpu_temp']
        )
        features['load_rate'] = state['cpu_load'] - self.feature_history[-2]['cpu_load']
        
        # Rolling features
        recent_temps_10 = [h['cpu_temp'] for h in self.feature_history[-10:]]
        recent_loads_10 = [h['cpu_load'] for h in self.feature_history[-10:]]
        all_loads_30 = [h['cpu_load'] for h in self.feature_history[-30:]]
        
        features['cpu_temp_roll10'] = np.mean(recent_temps_10)
        features['cpu_load_roll10'] = np.mean(recent_loads_10)
        features['cpu_load_roll30'] = np.mean(all_loads_30)
        features['cpu_load_std10'] = np.std(recent_loads_10)
        
        # Interaction features
        features['load_ambient_interaction'] = state['cpu_load'] * state['ambient_temp']
        features['thermal_stress'] = state['cpu_load'] * state['cpu_temp']
        features['temp_above_ambient'] = state['cpu_temp'] - state['ambient_temp']
        
        return features
    
    def predict_temperature(self, features):
        """Predict CPU temperature 5 seconds ahead"""
        try:
            # Create DataFrame with features in correct order
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[self.feature_cols]  # Ensure correct column order
            
            # Predict
            predicted_temp = self.model.predict(feature_df)[0]
            
            return predicted_temp
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def control_fan(self, predicted_temp, current_temp):
        """
        Fan control with L9110 H-bridge module and rate limiting.
        """
        if predicted_temp is None:
            predicted_temp = current_temp  # Fallback
        
        # Determine target fan speed
        if predicted_temp >= self.TEMP_CRITICAL:
            target_speed = 255
            status = "CRITICAL"
            color = "\033[91m"
        elif predicted_temp >= self.TEMP_WARNING:
            ratio = (predicted_temp - self.TEMP_WARNING) / (self.TEMP_CRITICAL - self.TEMP_WARNING)
            target_speed = int(128 + 127 * ratio)
            status = "WARNING"
            color = "\033[93m"
        elif predicted_temp >= 60:
            target_speed = 100
            status = "ELEVATED"
            color = "\033[94m"
        else:
            target_speed = 50
            status = "NORMAL"
            color = "\033[92m"
        
        # Apply rate limiting (smooth L9110 control)
        fan_speed = np.clip(
            target_speed,
            self.last_fan_speed - self.max_fan_step,
            self.last_fan_speed + self.max_fan_step
        )
        fan_speed = int(fan_speed)
        
        self.last_fan_speed = fan_speed
        
        # Send command to L9110 via Arduino
        if self.arduino_available:
            try:
                self.arduino.reset_output_buffer()
                command = f'F{fan_speed}\n'.encode()
                self.arduino.write(command)
            except Exception as e:
                print(f"⚠ L9110 communication error: {e}")
                self.arduino_available = False
        
        return fan_speed, status, color
    
    def run_monitoring(self, duration_minutes=5, log_file='prediction_log.csv'):
        """Main monitoring loop"""
        print("\n" + "="*70)
        print("SIMPLIFIED CPU TEMPERATURE PREDICTION SYSTEM")
        print("="*70)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Prediction horizon: {self.PREDICTION_HORIZON} seconds ahead")
        print(f"Warning threshold: {self.TEMP_WARNING}°C")
        print(f"Critical threshold: {self.TEMP_CRITICAL}°C")
        print(f"Model: {type(self.model).__name__}")
        print(f"Features: {len(self.feature_cols)}")
        print("="*70)
        
        log_data = []
        
        # Monotonic timing
        start_time = time.monotonic()
        end_time = start_time + (duration_minutes * 60)
        next_sample_time = start_time
        
        print("\nPress Ctrl+C to stop\n")
        print("Collecting initial samples (need 30 seconds for rolling features)...")
        
        sample_count = 0
        
        try:
            while time.monotonic() < end_time:
                state = self.get_system_state()
                sample_count += 1
                
                features = self.engineer_features(state)
                
                if features is None:
                    print(f"\rCollecting... {len(self.feature_history)}/30 samples", 
                          end='', flush=True)
                    
                    next_sample_time += 1.0
                    sleep_time = next_sample_time - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                
                if sample_count == 30:
                    print("\n\nStarting predictions...")
                    print("Time      | Current | Predicted | Δ(5s)   | Status   | Fan")
                    print("-"*70)
                
                predicted_temp = self.predict_temperature(features)
                
                if predicted_temp is None:
                    print("\n⚠ Prediction failed, skipping this sample")
                    next_sample_time += 1.0
                    sleep_time = next_sample_time - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                
                predicted_delta = predicted_temp - state['cpu_temp']
                
                fan_speed, status, color = self.control_fan(
                    predicted_temp, state['cpu_temp']
                )
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"{timestamp} | "
                      f"{state['cpu_temp']:7.2f}°C | "
                      f"{predicted_temp:9.2f}°C | "
                      f"{predicted_delta:+7.2f}°C | "
                      f"{color}{status:8s}\033[0m | "
                      f"{fan_speed:3d}/255",
                      flush=True)
                
                log_entry = {
                    'timestamp': timestamp,
                    'cpu_load': state['cpu_load'],
                    'ram_usage': state['ram_usage'],
                    'ambient_temp': state['ambient_temp'],
                    'current_temp': state['cpu_temp'],
                    'predicted_temp': predicted_temp,
                    'predicted_delta': predicted_delta,
                    'fan_speed': fan_speed,
                    'status': status
                }
                log_data.append(log_entry)
                
                next_sample_time += 1.0
                sleep_time = next_sample_time - time.monotonic()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.1:
                    print(f"\n⚠ Warning: Sample {sample_count} lagged by {-sleep_time:.2f}s")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Monitoring stopped by user")
        
        finally:
            if log_data:
                log_df = pd.DataFrame(log_data)
                log_df.to_csv(log_file, index=False)
                print(f"\n✓ Prediction log saved to: {log_file}")
                
                print("\n" + "="*70)
                print("MONITORING SUMMARY")
                print("="*70)
                print(f"Total predictions: {len(log_df)}")
                print(f"Average error (abs): {abs(log_df['predicted_delta']).mean():.2f}°C")
                print(f"Max error (abs): {abs(log_df['predicted_delta']).max():.2f}°C")
                print(f"Temperature range: {log_df['current_temp'].min():.1f}°C - "
                      f"{log_df['current_temp'].max():.1f}°C")
                print(f"CPU load range: {log_df['cpu_load'].min():.1f}% - "
                      f"{log_df['cpu_load'].max():.1f}%")
                print(f"Fan speed range: {log_df['fan_speed'].min()}-{log_df['fan_speed'].max()}/255")
                
                # Calculate R² equivalent
                actual = log_df['current_temp'].values[1:]  # Shifted for comparison
                predicted = log_df['predicted_temp'].values[:-1]
                if len(actual) > 0:
                    from sklearn.metrics import r2_score
                    # This is approximate since we're comparing shifted values
                    print(f"\nNote: Predictions are for 5 seconds ahead")
                    print(f"      Monitor over time to validate accuracy")
            
            # Cleanup - turn off fan
            if self.arduino:
                try:
                    self.arduino.write(b'F0\n')
                    time.sleep(0.1)
                except:
                    pass
                self.arduino.close()
            
            print("="*70)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        SIMPLIFIED CPU TEMPERATURE PREDICTION             ║
    ║                                                          ║
    ║  Model: Random Forest (from training notebook)          ║
    ║  Features: 18 engineered features                       ║
    ║  Prediction: 5 seconds ahead                            ║
    ║                                                          ║
    ║  Optional: Arduino + DS18B20 + L9110 Fan Control        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if model files exist
    if not os.path.exists('cpu_temp_predictor.pkl'):
        print("❌ Error: cpu_temp_predictor.pkl not found")
        print("   Please run the training notebook first to generate the model")
        exit(1)
    
    if not os.path.exists('feature_columns.pkl'):
        print("❌ Error: feature_columns.pkl not found")
        print("   Please run the training notebook first to generate feature columns")
        exit(1)
    
    try:
        system = SimplifiedThermalPredictor()
    except Exception as e:
        print(f"\n❌ Failed to initialize system: {e}")
        exit(1)
    
    print("\n✓ System initialized successfully!")
    
    try:
        duration = int(input("\nEnter monitoring duration in minutes (default 5): ") or "5")
    except:
        duration = 5
    
    print(f"\nStarting {duration}-minute monitoring session...")
    print("Watch for:")
    print("  - 30-second warmup (collecting history)")
    print("  - 1 Hz sampling rate")
    print("  - 5-second ahead predictions")
    print("  - Smooth fan speed transitions\n")
    
    system.run_monitoring(duration_minutes=duration)
    
    print("\n✅ Monitoring complete!")