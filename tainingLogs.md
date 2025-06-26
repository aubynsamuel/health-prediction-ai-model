| Class    | Oxygen (%)       | Heart Rate (bpm) |
| -------- | ---------------- | ---------------- |
| Healthy  | 95–100           | 60–100           |
| Warning  | 90–94 or 101–110 | 50–59 or 101–110 |
| Critical | <90              | <50 or >110      |

# Training Logs

⚗️ Engineering advanced medical features...
📊 Dataset: 12000 samples with 14 engineered features
🤖 Training advanced AI ensemble...
🎯 Cross-validation accuracy: 0.9871 (±0.0042)

✅ Validation Accuracy: 98.38%

📋 Detailed Classification Report:
precision recall f1-score support

     Healthy       1.00      0.99      1.00      1560
     Warning       0.95      0.98      0.97       600
    Critical       0.98      0.92      0.95       240

    accuracy                           0.98      2400

macro avg 0.98 0.97 0.97 2400
weighted avg 0.98 0.98 0.98 2400

📈 Training Data Distribution:
Healthy: 7,800 samples (65.0%)
Warning: 3,000 samples (25.0%)
Critical: 1,200 samples (10.0%)

🔍 Top AI-Selected Features:
SpO2 : 0.1880 █████████
SpO2² : 0.1767 ████████
O2_Deficit : 0.1744 ████████
Risk_Score : 0.1563 ███████
HR_O2_Stress: 0.1242 ██████
SpO2_Dev : 0.0957 ████
HR² : 0.0472 ██
HR_Dev : 0.0209 █
SpO2_Low : 0.0127
SpO2_Critical: 0.0037

💾 AI model package saved to: health_prediction_model.tflite
🎉 Model ready for deployment!
