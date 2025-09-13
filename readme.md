<div align="center">

# Health Monitoring AI for Wearables

**A lightweight machine learning model for real-time health monitoring on microcontrollers.**

</div>

This project provides a complete pipeline for training and deploying a TensorFlow Lite model that classifies health status based on Heart Rate (HR) and Blood Oxygen (SpO2) levels. It is designed for high efficiency and accuracy, making it ideal for wearable devices and embedded systems.

---

## ✨ Key Features

- **Dual-Output Model**: Predicts both SpO2 and HR status simultaneously, achieving >98% accuracy.
- **Optimized for Microcontrollers**: Fully INT8 quantized, resulting in a model size of less than 3 KB.
- **Professional Codebase**: Features modular code, type hinting, extensive logging, and clear configuration management.
- **Realistic Synthetic Data**: Includes a sophisticated data generation script that mimics real-world physiological patterns.
- **Interactive Inference**: Comes with a user-friendly CLI for real-time model testing and evaluation.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pip for package management

### 1. Clone & Setup

Clone the repository and install the required dependencies:

```bash
# It is recommended to use a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `\.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script to generate the `hrSpO2model.tflite` file:

```bash
python spo2hrmodeltraining.py
```

The script will log the entire training process, including data generation, model compilation, training, and evaluation.

### 3. Test the Model

Test the generated model with the interactive inference script:

#### Interactive Mode

```bash
python inference.py --interactive
```

You will be prompted to enter HR and SpO2 values.

#### Single Prediction

```bash
python inference.py --hr 75 --spo2 98
```

## 🤖 Model Details

- **Architecture**: A dual-output sequential model with shared hidden layers (32 and 24 neurons) using ReLU activation, L2 regularization, and dropout. The model then branches out to two separate heads for SpO2 and HR prediction, each with a 16-neuron hidden layer.
- **Output Layer**: A concatenated output of two softmax layers, providing probabilities for the health classes of each vital sign.
- **SpO2 Classes**:
  - **🟢 Normal**: 95-100%
  - **🟡 Mild Hypoxemia**: 90-94%
  - **🔴 High Hypoxemia**: <90%
- **Heart Rate Classes**:
  - **🟢 Normal**: 60-100 bpm
  - **🟡 Bradycardia**: <60 bpm
  - **🔴 Tachycardia**: >100 bpm

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
