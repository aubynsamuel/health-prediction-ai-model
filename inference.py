import numpy as np
import tensorflow as tf
import argparse

class HealthMonitorInference:
    def __init__(self, model_path):
        """
        Initialize the health monitor with a trained TFLite model.

        Args:
            model_path (str): Path to the TFLite model file
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        # Status labels matching Arduino code
        self.spo2_status = ["Normal", "Mild Hypoxemia", "High Hypoxemia"]
        self.hr_status = ["Normal", "Bradycardia", "Tachycardia"]

        self._load_model()

    def _load_model(self):
        """Load the TFLite model and get input/output details."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"✅ Model loaded successfully from: {self.model_path}")
            print(f"📊 Input shape: {self.input_details[0]['shape']}")
            print(f"📊 Output shape: {self.output_details[0]['shape']}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _normalize_inputs(self, heart_rate, spo2):
        """
        Robust normalization using physiological ranges:
        HR: 30-150 bpm -> [0, 1]
        SpO2: 80-100% -> [0, 1]
        """
        hr_norm = (heart_rate - 30.0) / 120.0
        spo2_norm = (spo2 - 80.0) / 20.0

        # Clip to ensure [0,1] range
        hr_norm = np.clip(hr_norm, 0.0, 1.0)
        spo2_norm = np.clip(spo2_norm, 0.0, 1.0)

        return np.array([[hr_norm, spo2_norm]], dtype=np.float32)

    def _quantize_input(self, input_data):
        """Quantize float32 input to uint8 based on interpreter's scale and zero_point."""
        input_details = self.input_details[0]
        input_scale, input_zero_point = input_details['quantization']
        quantized_input = input_data / input_scale + input_zero_point
        return np.array(quantized_input, dtype=input_details['dtype'])

    def _dequantize_output(self, output_data):
        """Dequantize uint8 output to float32 based on interpreter's scale and zero_point."""
        output_details = self.output_details[0]
        output_scale, output_zero_point = output_details['quantization']
        dequantized_output = (output_data - output_zero_point) * output_scale
        return dequantized_output


    def predict(self, heart_rate, spo2, verbose=True):
        """
        Make a prediction for the given heart rate and SpO2.

        Args:
            heart_rate (float): Heart rate in BPM
            spo2 (float): Oxygen saturation percentage
            verbose (bool): Whether to print detailed results

        Returns:
            dict: Dictionary containing predictions and probabilities
        """
        # Validate inputs
        if not (30 <= heart_rate <= 200):
            print("⚠️ Warning: Heart rate outside typical range (30-200 BPM)")

        if not (70 <= spo2 <= 100):
            print("⚠️ Warning: SpO2 outside typical range (70-100%)")

        # Normalize inputs
        input_data_float32 = self._normalize_inputs(heart_rate, spo2)

        # Quantize input data
        input_data_quantized = self._quantize_input(input_data_float32)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data_quantized)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data_quantized = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize output data
        output_data_float32 = self._dequantize_output(output_data_quantized)


        # Split outputs (first 3 for SpO2, last 3 for HR)
        spo2_probs = output_data_float32[0, 0:3]
        hr_probs = output_data_float32[0, 3:6]

        # Apply softmax to get probabilities for each task
        spo2_probs = tf.nn.softmax(spo2_probs).numpy()
        hr_probs = tf.nn.softmax(hr_probs).numpy()


        # Get predictions
        spo2_class = np.argmax(spo2_probs)
        hr_class = np.argmax(hr_probs)

        spo2_confidence = spo2_probs[spo2_class] * 100
        hr_confidence = hr_probs[hr_class] * 100

        results = {
            'heart_rate': heart_rate,
            'spo2': spo2,
            'spo2_status': self.spo2_status[spo2_class],
            'hr_status': self.hr_status[hr_class],
            'spo2_confidence': spo2_confidence,
            'hr_confidence': hr_confidence,
            'spo2_probabilities': spo2_probs,
            'hr_probabilities': hr_probs
        }

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results):
        """Print formatted results."""
        print("\n" + "="*50)
        print("🏥 HEALTH STATUS PREDICTION")
        print("="*50)
        print(f"📊 Heart Rate: {results['heart_rate']} BPM")
        print(f"📊 SpO2: {results['spo2']}%")
        print()
        print(f"💓 HR Status: {results['hr_status']} ({results['hr_confidence']:.1f}% confidence)")
        print(f"🫁 SpO2 Status: {results['spo2_status']} ({results['spo2_confidence']:.1f}% confidence)")

        # Show all probabilities
        print("\n📈 Detailed Probabilities:")
        print("SpO2 Classification:")
        for i, (status, prob) in enumerate(zip(self.spo2_status, results['spo2_probabilities'])):
            marker = "→" if i == np.argmax(results['spo2_probabilities']) else " "
            print(f"  {marker} {status}: {prob*100:.1f}%")

        print("Heart Rate Classification:")
        for i, (status, prob) in enumerate(zip(self.hr_status, results['hr_probabilities'])):
            marker = "→" if i == np.argmax(results['hr_probabilities']) else " "
            print(f"  {marker} {status}: {prob*100:.1f}%")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Health Monitor Inference")
    parser.add_argument("--model", type=str, default="hrSpO2model.tflite",
                       help="Path to TFLite model file")
    parser.add_argument("--hr", type=float, help="Heart rate in BPM")
    parser.add_argument("--spo2", type=float, help="SpO2 percentage")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    # Parse known arguments and ignore the rest
    args, unknown = parser.parse_known_args()

    # Initialize the health monitor
    try:
        monitor = HealthMonitorInference(args.model)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    if args.interactive or (args.hr is None and args.spo2 is None):
        # Interactive mode
        print("\n🏥 Interactive Health Monitor")
        print("Enter 'quit' or 'exit' to stop\n")

        while True:
            try:
                hr_input = input("Enter heart rate (BPM) or 'quit': ").strip()
                if hr_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                print(f"Debug: Received HR input: '{hr_input}'") # Debug print
                try:
                    hr = float(hr_input) # Attempt to convert HR input to float
                except ValueError:
                    print("❌ Invalid input for Heart Rate. Please enter a numerical value.")
                    continue # Skip to the next iteration

                spo2_input = input("Enter SpO2 (%) or 'quit': ").strip()
                if spo2_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                print(f"Debug: Received SpO2 input: '{spo2_input}'") # Debug print
                try:
                    spo2 = float(spo2_input) # Attempt to convert SpO2 input to float
                except ValueError:
                    print("❌ Invalid input for SpO2. Please enter a numerical value.")
                    continue # Skip to the next iteration


                # Make prediction
                monitor.predict(hr, spo2)
                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An unexpected error occurred: {e}")

    else:
        # Single prediction mode
        if args.hr is None or args.spo2 is None:
            print("❌ Please provide both --hr and --spo2 arguments")
            return

        monitor.predict(args.hr, args.spo2)


if __name__ == "__main__":
    main()
