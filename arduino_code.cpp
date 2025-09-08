#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "hrSpO2model.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
MAX30105 particleSensor;

#define SENSOR_BUFFER_SIZE 32
uint32_t irBuffer[SENSOR_BUFFER_SIZE];
uint32_t redBuffer[SENSOR_BUFFER_SIZE];

int32_t spo2;
int8_t validSPO2;
int32_t heartRate;
int8_t validHeartRate;

long hrSum = 0;
int sampleCount = 0;
unsigned long lastAvgTime = 0;
int32_t bestSpO2 = 0;
unsigned long lastSpO2UpdateTime = 0;

// TensorFlow Lite variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const char* spo2Status[] = {"Normal", "Mild Hypoxemia", "High Hypoxemia"};
const char* hrStatus[] = {"Normal", "Bradycardia", "Tachycardia"};

void setupTensorFlow() {
  model = tflite::GetModel(hrSpO2model_tflite);
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}

struct HealthStatusResult {
  int spo2Class;
  int hrClass;
};

HealthStatusResult predictHealthStatus(float hr, float spo2) {
  if (interpreter == nullptr) {
    setupTensorFlow();
  }

  float hr_norm = (hr - 30.0) / 120.0;
  float spo2_norm = (spo2 - 80.0) / 20.0;
  hr_norm = constrain(hr_norm, 0.0, 1.0);
  spo2_norm = constrain(spo2_norm, 0.0, 1.0);

  input->data.f[0] = hr_norm;
  input->data.f[1] = spo2_norm;
  interpreter->Invoke();

  HealthStatusResult result;
  
  // SpO2 classification based on first 3 output nodes
  float max_spo2_prob = -1.0;
  int spo2_class = 0;
  for (int i = 0; i < 3; ++i) {
    if (output->data.f[i] > max_spo2_prob) {
      max_spo2_prob = output->data.f[i];
      spo2_class = i;
    }
  }

  // Heart Rate classification based on next 3 output nodes
  float max_hr_prob = -1.0;
  int hr_class = 0;
  for (int i = 3; i < 6; ++i) {
    if (output->data.f[i] > max_hr_prob) {
      max_hr_prob = output->data.f[i];
      hr_class = i - 3;
    }
  }

  result.spo2Class = spo2_class;
  result.hrClass = hr_class;

  return result;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(4, 5);
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println(F("MAX30102 not found"));
    while (1);
  }

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 failed"));
    for (;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("HR/SpO2 Monitor");
  display.display();

  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);

  lastAvgTime = millis();
  lastSpO2UpdateTime = millis();
  bestSpO2 = 0;
}

void loop() {
  for (byte i = 0; i < SENSOR_BUFFER_SIZE; i++) {
    while (!particleSensor.available()) {
      particleSensor.check();
    }
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
  }

  long avgIR = 0;
  for (int i = 0; i < SENSOR_BUFFER_SIZE; i++) avgIR += irBuffer[i];
  avgIR /= SENSOR_BUFFER_SIZE;
  if (avgIR < 5000) {
    display.clearDisplay();
    display.setCursor(0, 20);
    display.println("No finger detected");
    display.display();

    bestSpO2 = 0;
    lastSpO2UpdateTime = millis();
    hrSum = 0;
    sampleCount = 0;
    lastAvgTime = millis();

    delay(1000);
    return;
  }

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer, SENSOR_BUFFER_SIZE,
      redBuffer, &spo2, &validSPO2,
      &heartRate, &validHeartRate);
  if (validHeartRate && heartRate > 30 && heartRate < 220) {
    hrSum += heartRate;
    sampleCount++;
  }
  if (validSPO2 && spo2 > 70 && spo2 <= 100) {
    if (spo2 > bestSpO2) bestSpO2 = spo2;
  }

  if (millis() - lastAvgTime >= 10000) {
    int avgHR = (sampleCount > 0) ?
    hrSum / sampleCount : -1;

    hrSum = 0;
    sampleCount = 0;
    lastAvgTime = millis();

    int32_t spo2ToDisplay;
    if (millis() - lastSpO2UpdateTime >= 40000) {
      spo2ToDisplay = bestSpO2;
      bestSpO2 = 0;
      lastSpO2UpdateTime = millis();
    } else {
      spo2ToDisplay = bestSpO2;
    }

    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("HR: ");
    if (avgHR > 0) {
      display.print(avgHR);
      display.println(" bpm");
    } else {
      display.println("--- bpm");
    }

    display.setCursor(0, 20);
    display.print("SpO2: ");
    if (spo2ToDisplay > 0) {
      display.print(spo2ToDisplay);
      display.println(" %");
    } else {
      display.println("--- %");
    }

    if (avgHR > 0 && spo2ToDisplay > 0) {
      HealthStatusResult status = predictHealthStatus(avgHR, spo2ToDisplay);

      display.setCursor(0, 40);
      display.print("HR Status: ");
      display.println(hrStatus[status.hrClass]);

      display.setCursor(0, 55);
      display.print("SpO2 Status: ");
      display.println(spo2Status[status.spo2Class]);
    }
    display.display();
  }
}
