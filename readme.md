<div align="center">

# Health Monitoring AI for Wearables

**A professional, lightweight machine learning model for real-time health monitoring on microcontrollers.**

</div>

This project provides a complete pipeline for training and deploying a TensorFlow Lite model that classifies health status based on Heart Rate (HR) and Blood Oxygen (SpO2) levels. It is designed for high efficiency and accuracy, making it ideal for wearable devices and embedded systems.

---

## ✨ Key Features

- **High-Performance Model**: Achieves >98% accuracy on test data.
- **Optimized for Microcontrollers**: Fully INT8 quantized, resulting in a model size of less than 3 KB.
- **Professional Codebase**: Features modular code, type hinting, extensive logging, and clear configuration management.
- **Realistic Synthetic Data**: Includes a sophisticated data generation script that mimics real-world physiological patterns.
- **Interactive Testing**: Comes with a user-friendly CLI for real-time model testing and evaluation.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pip for package management

### 1. Clone & Setup

Clone the repository and install the required dependencies:

```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `\.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script to generate the `health_model_optimized.tflite` file:

```bash
python train_model.py
```

The script will log the entire training process, including data generation, model compilation, training, and evaluation.

### 3. Test the Model

Test the generated model with the interactive testing script:

```bash
python test_model.py
```

You can input custom HR and SpO2 values or run a set of predefined test cases to see the model in action.

## 🤖 Model Details

- **Architecture**: A sequential model with two dense hidden layers (16 and 8 neurons respectively) using ReLU activation, L2 regularization, and dropout to prevent overfitting.
- **Output Layer**: A dense layer with 3 neurons and a softmax activation function to output probabilities for the three health classes.
- **Classes**:
  - **🟢 Healthy**: Normal vital signs.
  - **🟡 Warning**: Vitals are outside the ideal range, indicating potential risk.
  - **🔴 Critical**: Vitals are in a dangerous range, requiring immediate attention.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature/YourFeature`)
3. **Commit your changes** (`git commit -m 'Add some feature'`)
4. **Push to the branch** (`git push origin feature/YourFeature`)
5. **Open a pull request**

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
