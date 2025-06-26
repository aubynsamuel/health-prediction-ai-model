import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_realistic_synthetic_data(n_samples=50000, random_seed=42):
    """
    Generate more realistic HR and SpO2 data with physiological correlations.
    Uses multiple distributions to simulate different health conditions.
    
    Classification:
    - Critical (2): SpO2 < 90 OR HR < 50 OR HR > 120
    - Warning (1): (90 ≤ SpO2 < 95) OR (50 ≤ HR < 60) OR (100 < HR ≤ 120)  
    - Healthy (0): SpO2 ≥ 95 AND 60 ≤ HR ≤ 100
    """
    np.random.seed(random_seed)
    
    # Generate samples for each class with realistic distributions
    n_healthy = int(n_samples * 0.6)    # 60% healthy
    n_warning = int(n_samples * 0.25)   # 25% warning
    n_critical = int(n_samples * 0.15)  # 15% critical
    
    hr_data = []
    spo2_data = []
    labels = []
    
    # Healthy samples (normal distributions around healthy ranges)
    for _ in range(n_healthy):
        # Healthy HR: 60-100, centered around 75
        hr = np.random.normal(75, 12)
        hr = np.clip(hr, 60, 100)
        
        # Healthy SpO2: 95-100, centered around 98
        spo2 = np.random.normal(98, 1.5)
        spo2 = np.clip(spo2, 95, 100)
        
        # Add some correlation (lower HR often with higher SpO2 in healthy individuals)
        if np.random.random() < 0.3:  # 30% correlation
            if hr < 70:
                spo2 = np.random.normal(99, 0.8)
                spo2 = np.clip(spo2, 96, 100)
        
        hr_data.append(hr)
        spo2_data.append(spo2)
        labels.append(0)  # Healthy
    
    # Warning samples
    for _ in range(n_warning):
        choice = np.random.choice(['low_spo2', 'low_hr', 'high_hr'])
        
        if choice == 'low_spo2':
            # SpO2 in warning range: 90-94
            spo2 = np.random.uniform(90, 95)
            hr = np.random.normal(75, 15)  # Can have various HR
            hr = np.clip(hr, 55, 110)
        elif choice == 'low_hr':
            # HR in warning range: 50-59
            hr = np.random.uniform(50, 60)
            spo2 = np.random.normal(96, 2)  # Usually decent SpO2
            spo2 = np.clip(spo2, 92, 100)
        else:  # high_hr
            # HR in warning range: 101-120
            hr = np.random.uniform(100, 121)
            spo2 = np.random.normal(96, 2)
            spo2 = np.clip(spo2, 93, 100)
        
        hr_data.append(hr)
        spo2_data.append(spo2)
        labels.append(1)  # Warning
    
    # Critical samples
    for _ in range(n_critical):
        choice = np.random.choice(['very_low_spo2', 'very_low_hr', 'very_high_hr', 'combined'])
        
        if choice == 'very_low_spo2':
            # Critical SpO2: < 90
            spo2 = np.random.uniform(80, 90)
            hr = np.random.normal(80, 20)  # Often elevated in hypoxia
            hr = np.clip(hr, 45, 140)
        elif choice == 'very_low_hr':
            # Critical HR: < 50
            hr = np.random.uniform(30, 50)
            spo2 = np.random.normal(94, 4)  # Variable SpO2
            spo2 = np.clip(spo2, 85, 100)
        elif choice == 'very_high_hr':
            # Critical HR: > 120
            hr = np.random.uniform(120, 150)
            spo2 = np.random.normal(94, 4)
            spo2 = np.clip(spo2, 85, 100)
        else:  # combined critical
            # Multiple critical factors
            spo2 = np.random.uniform(80, 92)
            hr = np.random.uniform(45, 130)
        
        hr_data.append(hr)
        spo2_data.append(spo2)
        labels.append(2)  # Critical
    
    # Add noise and edge cases
    noise_samples = int(n_samples * 0.05)
    for _ in range(noise_samples):
        hr = np.random.uniform(30, 150)
        spo2 = np.random.uniform(80, 100)
        
        # Classify based on rules
        if spo2 < 90 or hr < 50 or hr > 120:
            label = 2
        elif (90 <= spo2 < 95) or (50 <= hr < 60) or (100 < hr <= 120):
            label = 1
        else:
            label = 0
            
        hr_data.append(hr)
        spo2_data.append(spo2)
        labels.append(label)
    
    # Convert to arrays
    X_raw = np.column_stack([np.array(hr_data), np.array(spo2_data)])
    y = np.array(labels)
    
    # Shuffle the data
    indices = np.random.permutation(len(y))
    X_raw = X_raw[indices]
    y = y[indices]
    
    print(f"Generated {len(y)} samples:")
    print(f"  Healthy: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Warning: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Critical: {np.sum(y == 2)} ({np.sum(y == 2)/len(y)*100:.1f}%)")
    
    return X_raw.astype(np.float32), y.astype(np.int32)

def normalize_features(X_raw):
    """
    Robust normalization using physiological ranges:
    HR: 30-150 bpm -> [0, 1]
    SpO2: 80-100% -> [0, 1]
    """
    hr = X_raw[:, 0]
    spo2 = X_raw[:, 1]
    
    hr_norm = (hr - 30.0) / 120.0
    spo2_norm = (spo2 - 80.0) / 20.0
    
    # Clip to ensure [0,1] range
    hr_norm = np.clip(hr_norm, 0, 1)
    spo2_norm = np.clip(spo2_norm, 0, 1)
    
    return np.column_stack([hr_norm, spo2_norm]).astype(np.float32)

def build_optimized_model():
    """
    Build a compact but effective model optimized for microcontrollers.
    Uses fewer parameters while maintaining good performance.
    """
    model = keras.Sequential([
        layers.Input(shape=(2,), name='input'),
        
        # First hidden layer - larger to capture feature interactions
        layers.Dense(24, activation='relu', name='dense1',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        
        # Second hidden layer - smaller
        layers.Dense(16, activation='relu', name='dense2',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(3, activation='softmax', name='output')
    ])
    
    # Use a lower learning rate for better convergence
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def representative_dataset_generator(X_norm):
    """
    Generator for quantization calibration.
    """
    for i in range(min(200, X_norm.shape[0])):
        sample = X_norm[i:i+1]
        yield [sample]

def convert_to_tflite_int8(keras_model, X_norm, tflite_path):
    """
    Convert to optimized TFLite model for microcontrollers.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Optimize for size and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_generator(X_norm)
    
    # Full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    try:
        tflite_model = converter.convert()
        
        # Save model
        os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / 1024  # Size in KB
        print(f"✅ TFLite model saved to: {tflite_path}")
        print(f"📏 Model size: {model_size:.2f} KB")
        
        return tflite_model
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return None

def evaluate_model_detailed(model, X_val, y_val):
    """
    Detailed model evaluation with metrics.
    """
    # Get predictions
    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_val)
    
    print(f"\n📊 Detailed Evaluation Results:")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    for class_idx in range(3):
        class_name = ['Healthy', 'Warning', 'Critical'][class_idx]
        class_mask = y_val == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == class_idx)
            print(f"{class_name} Accuracy: {class_acc*100:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    target_names = ['Healthy', 'Warning', 'Critical']
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    return accuracy

def main(args):
    print("🏥 Training Improved Health Monitoring Model")
    print("=" * 50)
    
    # 1) Generate realistic synthetic data
    print("🔄 Generating realistic synthetic data...")
    X_raw, y = generate_realistic_synthetic_data(
        n_samples=args.samples, 
        random_seed=42
    )
    
    # 2) Normalize features
    print("⚙️ Normalizing features...")
    X_norm = normalize_features(X_raw)
    
    # 3) Split data (train/val/test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # 4) Build and train model
    print("\n🏗️ Building optimized model...")
    model = build_optimized_model()
    model.summary()
    
    # Calculate model parameters
    total_params = model.count_params()
    print(f"📊 Total parameters: {total_params:,}")
    
    print("\n🚀 Training model...")
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5) Evaluate model
    print("\n📈 Evaluating model performance...")
    val_accuracy = evaluate_model_detailed(model, X_val, y_val)
    
    # Test set evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Final Test Accuracy: {test_acc*100:.2f}%")
    
    # 6) Convert to TFLite
    print("\n⚙️ Converting to TFLite for microcontroller...")
    tflite_model = convert_to_tflite_int8(model, X_train, args.tflite_path)
    
    if tflite_model:
        print("\n✅ Model training and conversion completed successfully!")
        print(f"📁 TFLite model saved as: {args.tflite_path}")
        print("🔧 Ready for deployment on Arduino/microcontroller!")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train improved health monitoring model for microcontrollers"
    )
    parser.add_argument(
        "--samples", type=int, default=50000,
        help="Number of synthetic samples (default: 50000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--tflite_path", type=str, default="health_model_optimized.tflite",
        help="Output TFLite file path"
    )
    
    args = parser.parse_args()
    model, history = main(args)