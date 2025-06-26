<div align="center">

# Health Monitoring TensorFlow Lite Model

A lightweight machine learning model for health monitoring using heart rate and SpO2 data, optimized for micro-controllers like Arduino.

</div>

## Features

- **Input**: Heart Rate (30-150 bpm) and SpO2 (80-100%)
- **Output**: Health status classification (Healthy, Warning, Critical)
- **Optimized**: INT8 quantized TFLite model (<10KB)
- **Ready**: For Arduino/ESP32 deployment

## Quick Start

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

Options:

- `--samples 50000` - Number of training samples
- `--epochs 100` - Training epochs

### 3. Test the Model

```bash
python test_model.py
```

Interactive testing with your own values or predefined test cases.

## Classification Rules

- **🟢 Healthy**: SpO2 ≥ 95% AND HR 60-100 bpm
- **🟡 Warning**: SpO2 90-94% OR HR 50-59 OR HR 101-120 bpm
- **🔴 Critical**: SpO2 < 90% OR HR < 50 OR HR > 120 bpm

## Model Architecture

- Input: 2 features (HR, SpO2)
- Hidden: 24 → 16 neurons with dropout
- Output: 3 classes (softmax)
- Size: ~8KB quantized INT8 model

## Files

- `train_model.py` - Model training with synthetic data generation
- `test_model.py` - Interactive model testing
- `health_model_optimized.tflite` - Generated TFLite model

## Usage Notes

- Model uses realistic synthetic data with physiological correlations
- Inputs are automatically normalized to [0,1] range
- INT8 quantization for microcontroller compatibility
- Achieves >95% accuracy on test data
