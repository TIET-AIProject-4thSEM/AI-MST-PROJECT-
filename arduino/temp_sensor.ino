/*
 * DS18B20 Temperature Sensor & L9110 Fan Control
 * ==============================================
 * Clean production code for thermal prediction system
 * 
 * Features:
 * - DS18B20 high-precision temperature reading (±0.5°C)
 * - L9110 PWM fan speed control (0-255)
 * - Works in Serial Monitor (manual testing)
 * - Works with Python script (automated control)
 * - Simple command protocol
 * 
 * Hardware:
 * - DS18B20 Temperature Sensor → Pin 2 (OneWire)
 * - L9110 Fan Module:
 *   → IA (Pin 9) - PWM speed control
 *   → IB (Pin 8) - Direction control
 * 
 * Commands:
 * - 'T' → Returns temperature (e.g., "24.5625")
 * - 'F<speed>' → Set fan speed 0-255 (e.g., "F128" = 50%)
 * - 'S' → System status (debug info)
 * 
 * Author: Thermal Prediction Project
 * Version: 4.0 - Clean & Simple
 */

#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================
#define ONE_WIRE_BUS 2    // DS18B20 data pin
#define FAN_IA 9          // L9110 speed control (PWM)
#define FAN_IB 8          // L9110 direction control

// Fan direction (adjust if fan spins wrong way)
#define FAN_REVERSE 1     // 1 = Reverse, 0 = Forward

// ============================================================================
// DS18B20 SETUP
// ============================================================================
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
DeviceAddress tempSensor;

// ============================================================================
// STATE TRACKING
// ============================================================================
int currentFanSpeed = 0;
bool ds18b20Ready = false;

// ============================================================================
// SETUP - RUN ONCE AT STARTUP
// ============================================================================
void setup() {
    // Start serial communication (9600 baud for reliability)
    Serial.begin(9600);
    
    // Give serial time to initialize
    delay(100);
    
    // ========================================
    // Initialize DS18B20 Temperature Sensor
    // ========================================
    sensors.begin();
    int deviceCount = sensors.getDeviceCount();
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("  Thermal Control System v4.0");
    Serial.println("  Hardware: DS18B20 + L9110");
    Serial.println("========================================");
    Serial.println();
    
    Serial.print("DS18B20 sensors detected: ");
    Serial.println(deviceCount);
    
    if (deviceCount > 0 && sensors.getAddress(tempSensor, 0)) {
        // Set to 12-bit resolution for max precision (0.0625°C)
        sensors.setResolution(tempSensor, 12);
        ds18b20Ready = true;
        
        Serial.println("✓ DS18B20 initialized");
        Serial.println("  Resolution: 12-bit (0.0625°C)");
        Serial.println("  Accuracy: ±0.5°C");
        
        // Print sensor address (for debugging)
        Serial.print("  Address: 0x");
        for (uint8_t i = 0; i < 8; i++) {
            if (tempSensor[i] < 16) Serial.print("0");
            Serial.print(tempSensor[i], HEX);
        }
        Serial.println();
    } else {
        Serial.println("✗ ERROR: DS18B20 not found!");
        Serial.println("  Check wiring:");
        Serial.println("    • VCC (Red) → 5V");
        Serial.println("    • GND (Black) → GND");
        Serial.println("    • DATA (Yellow) → Pin 2");
        Serial.println("    • 4.7kΩ resistor: VCC ↔ DATA");
    }
    
    // ========================================
    // Initialize L9110 Fan Controller
    // ========================================
    pinMode(FAN_IA, OUTPUT);
    pinMode(FAN_IB, OUTPUT);
    
    // Start fan at 50% for testing
    setFanSpeed(128);
    currentFanSpeed = 128;
    
    Serial.println("✓ L9110 initialized");
    Serial.println("  Initial speed: 50% (128/255)");
    #if FAN_REVERSE
    Serial.println("  Direction: REVERSE");
    #else
    Serial.println("  Direction: FORWARD");
    #endif
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("  READY - Waiting for commands");
    Serial.println("========================================");
    Serial.println();
    Serial.println("Commands:");
    Serial.println("  T       → Get temperature");
    Serial.println("  F<num>  → Set fan speed (0-255)");
    Serial.println("            Example: F192 = 75%");
    Serial.println("  S       → System status");
    Serial.println();
}

// ============================================================================
// MAIN LOOP - RUNS CONTINUOUSLY
// ============================================================================
void loop() {
    // Check for incoming commands
    if (Serial.available() > 0) {
        char command = Serial.read();
        
        // Ignore newline and carriage return
        if (command == '\n' || command == '\r') {
            return;
        }
        
        // Process command
        switch (command) {
            case 'T':
            case 't':
                handleTemperatureRequest();
                break;
                
            case 'F':
            case 'f':
                handleFanControl();
                break;
                
            case 'S':
            case 's':
                handleStatusRequest();
                break;
                
            default:
                Serial.print("ERROR: Unknown command '");
                Serial.print(command);
                Serial.println("'");
                Serial.println("Use: T (temp), F<num> (fan), S (status)");
                break;
        }
    }
    
    // Small delay to prevent serial buffer issues
    delay(10);
}

// ============================================================================
// COMMAND HANDLER: TEMPERATURE REQUEST
// ============================================================================
void handleTemperatureRequest() {
    /*
     * Read DS18B20 and return temperature in Celsius.
     * 
     * Response format:
     * - Success: "24.5625" (temperature with 4 decimals)
     * - Failure: "ERROR"
     * 
     * DS18B20 returns -127.00 on disconnection/error.
     */
    
    if (!ds18b20Ready) {
        Serial.println("ERROR");
        return;
    }
    
    // Request temperature reading from sensor
    sensors.requestTemperatures();
    
    // Read temperature in Celsius
    float temp = sensors.getTempC(tempSensor);
    
    // Validate reading
    // DS18B20 valid range: -55°C to +125°C
    if (temp == DEVICE_DISCONNECTED_C || temp < -55.0 || temp > 125.0) {
        Serial.println("ERROR");
    } else {
        // Send temperature with 4 decimal places (12-bit precision)
        Serial.println(temp, 4);
    }
}

// ============================================================================
// COMMAND HANDLER: FAN SPEED CONTROL
// ============================================================================
void handleFanControl() {
    /*
     * Set L9110 fan speed via PWM.
     * 
     * Command format: F<speed>
     * - F0   → Fan OFF
     * - F128 → 50% speed
     * - F255 → 100% speed
     * 
     * Response:
     * - Success: "OK: Fan = 128/255"
     * - Failure: "ERROR: Speed must be 0-255"
     */
    
    // Read speed value from serial buffer
    int speed = Serial.parseInt();
    
    // Validate speed range
    if (speed < 0 || speed > 255) {
        Serial.print("ERROR: Invalid speed ");
        Serial.print(speed);
        Serial.println(" (must be 0-255)");
        return;
    }
    
    // Set fan speed
    setFanSpeed(speed);//might comment out -- incase
    currentFanSpeed = speed;
    
    // Confirmation
    Serial.print("OK: Fan = ");
    Serial.print(speed);
    Serial.print("/255 (");
    Serial.print((speed * 100) / 255);
    Serial.println("%)");
}

// ============================================================================
// COMMAND HANDLER: SYSTEM STATUS
// ============================================================================
void handleStatusRequest() {
    /*
     * Display current system state for debugging.
     * Useful for Serial Monitor testing.
     */
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("         SYSTEM STATUS");
    Serial.println("========================================");
    
    // Temperature sensor status
    Serial.print("DS18B20 Sensor: ");
    if (!ds18b20Ready) {
        Serial.println("NOT DETECTED");
    } else {
        sensors.requestTemperatures();
        float temp = sensors.getTempC(tempSensor);
        
        if (temp == DEVICE_DISCONNECTED_C) {
            Serial.println("ERROR (disconnected)");
        } else {
            Serial.print("OK → ");
            Serial.print(temp, 2);
            Serial.println(" °C");
        }
    }
    
    // Fan status
    Serial.print("L9110 Fan Speed: ");
    Serial.print(currentFanSpeed);
    Serial.print("/255 (");
    Serial.print((currentFanSpeed * 100) / 255);
    Serial.println("%)");
    
    // Direction
    Serial.print("Fan Direction: ");
    #if FAN_REVERSE
    Serial.println("REVERSE");
    #else
    Serial.println("FORWARD");
    #endif
    
    // Uptime
    Serial.print("Uptime: ");
    unsigned long seconds = millis() / 1000;
    unsigned long minutes = seconds / 60;
    unsigned long hours = minutes / 60;
    
    if (hours > 0) {
        Serial.print(hours);
        Serial.print("h ");
    }
    Serial.print(minutes % 60);
    Serial.print("m ");
    Serial.print(seconds % 60);
    Serial.println("s");
    
    // Free memory (useful for debugging)
    Serial.print("Free RAM: ");
    Serial.print(freeMemory());
    Serial.println(" bytes");
    
    Serial.println("========================================");
    Serial.println();
}

// ============================================================================
// L9110 FAN SPEED CONTROL
// ============================================================================
void setFanSpeed(int speed) {
    if (speed == 0) {
        analogWrite(FAN_IA, 0);
        digitalWrite(FAN_IB, LOW);
        return;
    }

#if FAN_REVERSE
    digitalWrite(FAN_IB, HIGH);   // reverse direction
#else
    digitalWrite(FAN_IB, LOW);    // forward direction
#endif

    analogWrite(FAN_IA, speed);   // PWM ONLY on pin 9
}
// ============================================================================
// UTILITY: FREE MEMORY CHECK
// ============================================================================
int freeMemory() {
    /*
     * Returns approximate free RAM in bytes.
     * Useful for debugging memory leaks.
     * 
     * Arduino Uno has 2KB RAM total.
     * Typical usage: 200-400 bytes for this program.
     */
    extern int __heap_start, *__brkval;
    int v;
    return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}

/*
 * ============================================================================
 * HARDWARE WIRING GUIDE
 * ============================================================================
 * 
 * DS18B20 TEMPERATURE SENSOR:
 * ---------------------------
 * Pin        → Arduino
 * --------   ---------
 * VCC (Red)    → 5V
 * GND (Black)  → GND
 * DATA (Yellow)→ Pin 2
 * 
 * CRITICAL: Add 4.7kΩ pull-up resistor between VCC and DATA!
 * (Some modules have it built-in - check with multimeter)
 * 
 * 
 * L9110 FAN MODULE:
 * -----------------
 * Pin      → Arduino
 * ----     ---------
 * VCC      → 5V (or external 5-12V for powerful fans)
 * GND      → GND (common ground with Arduino!)
 * A-IA     → Pin 9 (PWM speed control)
 * A-IB     → Pin 8 (direction control)
 * 
 * Motor Connections:
 * MOTOR A+ → Fan positive wire (usually red)
 * MOTOR A- → Fan negative wire (usually black)
 * 
 * POWER NOTES:
 * - Small fans (<500mA): Arduino 5V is sufficient
 * - Large fans (>500mA): Use external 5-12V power supply
 *   → Connect external GND to Arduino GND (CRITICAL!)
 *   → Arduino provides control signals only
 * 
 * FAN DIRECTION:
 * - If fan spins wrong direction, change FAN_REVERSE from 1 to 0
 * - Or swap the motor wires (A+ ↔ A-)
 * 
 * ============================================================================
 * TESTING PROCEDURE
 * ============================================================================
 * 
 * 1. SERIAL MONITOR TESTING:
 * --------------------------
 * Open Arduino Serial Monitor (9600 baud, Both NL & CR)
 * 
 * Test temperature:
 *   Type: T
 *   Expected: "24.5625" (current room temperature)
 * 
 * Test fan speed:
 *   Type: F0     → Fan should stop
 *   Type: F128   → Fan at 50%
 *   Type: F255   → Fan at 100%
 * 
 * Check status:
 *   Type: S
 *   Expected: Full system status display
 * 
 * 
 * 2. PYTHON INTEGRATION TESTING:
 * ------------------------------
 * Use this Python snippet:
 * 
 *   import serial
 *   import time
 *   
 *   arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
 *   time.sleep(2)  # Wait for Arduino to initialize
 *   
 *   # Test temperature
 *   arduino.write(b'T\n')
 *   temp = arduino.readline().decode().strip()
 *   print(f"Temperature: {temp}°C")
 *   
 *   # Test fan
 *   arduino.write(b'F192\n')  # 75% speed
 *   response = arduino.readline().decode().strip()
 *   print(response)
 * 
 * 
 * 3. CONTINUOUS MONITORING:
 * -------------------------
 * Test repeated commands (like Python script does):
 * 
 *   while True:
 *       arduino.write(b'T\n')
 *       temp = arduino.readline()
 *       arduino.write(b'F128\n')
 *       time.sleep(1)
 * 
 * 
 * ============================================================================
 * TROUBLESHOOTING
 * ============================================================================
 * 
 * PROBLEM: DS18B20 returns "ERROR"
 * SOLUTION:
 *   1. Check wiring (especially DATA → Pin 2)
 *   2. Verify 4.7kΩ pull-up resistor present
 *   3. Check power supply (5V steady)
 *   4. Try different DS18B20 module (could be faulty)
 *   5. Measure resistance: DATA-to-VCC should be ~4.7kΩ
 * 
 * PROBLEM: Fan doesn't spin
 * SOLUTION:
 *   1. Check L9110 power (VCC should be 5-12V)
 *   2. Verify motor connections (A+ and A-)
 *   3. Try full speed: F255
 *   4. Check if fan needs minimum voltage (some need 30%+ to start)
 *   5. Swap FAN_REVERSE setting (try 0 and 1)
 *   6. Test with multimeter: Pin 9 should show ~5V at F255
 * 
 * PROBLEM: Serial commands not working
 * SOLUTION:
 *   1. Verify baud rate: 9600 in both Arduino and Serial Monitor
 *   2. Check line ending: "Both NL & CR" in Serial Monitor
 *   3. Try lowercase commands (t, f, s)
 *   4. Re-upload Arduino code
 *   5. Check USB cable (data, not just charging)
 * 
 * PROBLEM: Fan spins wrong direction
 * SOLUTION:
 *   1. Change #define FAN_REVERSE from 1 to 0 (or vice versa)
 *   2. Or swap motor wires: A+ ↔ A-
 *   3. Re-upload code
 * 
 * PROBLEM: Erratic temperature readings
 * SOLUTION:
 *   1. Add/check 4.7kΩ pull-up resistor
 *   2. Use shorter wire (< 3 meters)
 *   3. Add 0.1µF capacitor near DS18B20 (noise filtering)
 *   4. Keep wire away from power lines (EMI interference)
 * 
 * PROBLEM: Arduino resets when fan starts
 * SOLUTION:
 *   1. Power issue! Fan draws too much current
 *   2. Use external power supply for L9110
 *   3. Add 100µF capacitor across Arduino 5V-GND
 *   4. Use separate power supply for Arduino and fan
 * 
 * ============================================================================
 * PERFORMANCE SPECS
 * ============================================================================
 * 
 * DS18B20 Temperature Sensor:
 * - Accuracy: ±0.5°C (-10°C to +85°C)
 * - Resolution: 0.0625°C (12-bit)
 * - Range: -55°C to +125°C
 * - Conversion time: 750ms (12-bit)
 * - Interface: OneWire (single data line)
 * 
 * L9110 Motor Driver:
 * - Operating voltage: 2.5V - 12V
 * - Output current: 800mA per channel (continuous)
 * - Peak current: 1.5A per channel
 * - PWM frequency: Compatible with Arduino (490Hz / 980Hz)
 * - Logic voltage: 5V (Arduino compatible)
 * 
 * Arduino Timing:
 * - Loop frequency: ~100 Hz (10ms delay)
 * - Command response: < 5ms (instant)
 * - Temperature read: ~750ms (DS18B20 conversion)
 * 
 * ============================================================================
 */
