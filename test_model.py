"""
Simple tester for the health monitoring TensorFlow Lite model.
Input: Heart Rate and SpO2
Output: Healthy, Warning, or Critical
"""

import numpy as np
import tensorflow as tf

def load_model(model_path="health_model_optimized.tflite"):
    """Load the TensorFlow Lite model."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✅ Model loaded: {model_path}")
        return interpreter
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the model first!")
        return None

def get_user_input():
    """Get heart rate and SpO2 from user."""
    print("\n🏥 Health Monitor Test")
    print("-" * 30)
    
    while True:
        try:
            hr = float(input("💓 Enter Heart Rate (30-150): "))
            if 30 <= hr <= 150:
                break
            print("   Please enter a value between 30-150")
        except ValueError:
            print("   Please enter a valid number")
    
    while True:
        try:
            spo2 = float(input("🫁 Enter SpO2 (80-100): "))
            if 80 <= spo2 <= 100:
                break
            print("   Please enter a value between 80-100")
        except ValueError:
            print("   Please enter a valid number")
    
    return hr, spo2

def normalize_inputs(heart_rate, spo2):
    """Normalize inputs for UINT8 quantized model."""
    hr_norm = (heart_rate - 30.0) / 120.0
    spo2_norm = (spo2 - 80.0) / 20.0
    
    # Ensure values are in [0,1] range
    hr_norm = max(0, min(1, hr_norm))
    spo2_norm = max(0, min(1, spo2_norm))
    
    # Convert to UINT8 range [0, 255]
    hr_uint8 = int(hr_norm * 255)
    spo2_uint8 = int(spo2_norm * 255)
    
    return np.array([[hr_uint8, spo2_uint8]], dtype=np.uint8)

def predict_health(interpreter, heart_rate, spo2):
    """Make prediction using the TFLite model."""
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input data (UINT8 format)
    input_data = normalize_inputs(heart_rate, spo2)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get results (output is also UINT8, need to convert)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Convert UINT8 output [0,255] back to probabilities [0,1]
    probabilities = output_data[0].astype(np.float32) / 255.0
    
    # Normalize probabilities to sum to 1
    probabilities = probabilities / np.sum(probabilities)
    
    prediction = np.argmax(probabilities)
    confidence = probabilities[prediction]
    
    return prediction, probabilities, confidence

def display_result(prediction, probabilities, confidence, hr, spo2):
    """Display the prediction results."""
    labels = ["🟢 Healthy", "🟡 Warning", "🔴 Critical"]
    
    print(f"\n📊 Results for HR: {hr}, SpO2: {spo2}%")
    print("-" * 40)
    print(f"🎯 Prediction: {labels[prediction]}")
    print(f"🔍 Confidence: {confidence*100:.1f}%")
    
    print(f"\n📈 All Probabilities:")
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        bar = "█" * int(prob * 20)
        print(f"   {label}: {prob*100:5.1f}% |{bar}")

def test_sample_cases(interpreter):
    """Test with some predefined cases."""
    test_cases = [
        (75, 98, "Normal healthy person"),
        (45, 87, "Low HR + Low SpO2 (Critical)"),
        (110, 92, "High HR + Low SpO2 (Warning)"),
        (65, 89, "Borderline case"),
        (130, 85, "Very concerning case")
    ]
    
    print("\n🧪 Testing Sample Cases:")
    print("=" * 50)
    
    for hr, spo2, description in test_cases:
        print(f"\n📋 Test: {description}")
        prediction, probabilities, confidence = predict_health(interpreter, hr, spo2)
        
        labels = ["Healthy", "Warning", "Critical"]
        result = labels[prediction]
        print(f"   HR: {hr}, SpO2: {spo2}% → {result} ({confidence*100:.1f}%)")

def main():
    print("🤖 Health Model Tester")
    print("=" * 30)
    
    # Load model
    interpreter = load_model()
    if not interpreter:
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Test with your values")
        print("2. Run sample test cases")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            try:
                hr, spo2 = get_user_input()
                prediction, probabilities, confidence = predict_health(interpreter, hr, spo2)
                display_result(prediction, probabilities, confidence, hr, spo2)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            test_sample_cases(interpreter)
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("Please enter 1, 2, or 3")

if __name__ == "__main__":
    main()