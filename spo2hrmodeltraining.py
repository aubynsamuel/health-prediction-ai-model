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
    Generate realistic HR and SpO2 data with physiological correlations.

    SpO2 Classification:
    - Normal (0): 95-100%
    - Mild Hypoxemia (1): 90-94%
    - High Hypoxemia (2): <90%

    HR Classification:
    - Normal (0): 60-100 bpm
    - Bradycardia (1): <60 bpm
    - Tachycardia (2): >100 bpm
    """
    np.random.seed(random_seed)

    hr_data = []
    spo2_data = []
    spo2_labels = []
    hr_labels = []

    # Generate samples across all combinations of conditions
    for _ in range(n_samples):
        # Randomly choose SpO2 condition
        spo2_condition = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # Most samples normal

        if spo2_condition == 0:  # Normal SpO2 (95-100%)
            spo2 = np.random.normal(98, 1.5)
            spo2 = np.clip(spo2, 95, 100)
        elif spo2_condition == 1:  # Mild Hypoxemia (90-94%)
            spo2 = np.random.uniform(90, 95)
        else:  # High Hypoxemia (<90%)
            spo2 = np.random.uniform(80, 90)

        # Randomly choose HR condition with some correlation to SpO2
        if spo2_condition == 2:  # High hypoxemia often causes tachycardia
            hr_condition = np.random.choice([0, 1, 2], p=[0.3, 0.1, 0.6])
        elif spo2_condition == 1:  # Mild hypoxemia might cause slight tachycardia
            hr_condition = np.random.choice([0, 1, 2], p=[0.5, 0.15, 0.35])
        else:  # Normal SpO2
            hr_condition = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])

        if hr_condition == 0:  # Normal HR (60-100 bpm)
            hr = np.random.normal(75, 12)
            hr = np.clip(hr, 60, 100)
        elif hr_condition == 1:  # Bradycardia (<60 bpm)
            hr = np.random.uniform(35, 60)
        else:  # Tachycardia (>100 bpm)
            hr = np.random.uniform(100, 150)

        hr_data.append(hr)
        spo2_data.append(spo2)
        spo2_labels.append(spo2_condition)
        hr_labels.append(hr_condition)

    # Convert to arrays
    X_raw = np.column_stack([np.array(hr_data), np.array(spo2_data)])
    spo2_labels = np.array(spo2_labels)
    hr_labels = np.array(hr_labels)

    # Shuffle the data
    indices = np.random.permutation(len(spo2_labels))
    X_raw = X_raw[indices]
    spo2_labels = spo2_labels[indices]
    hr_labels = hr_labels[indices]

    print(f"Generated {len(spo2_labels)} samples:")
    print(f"SpO2 Distribution:")
    print(f"  Normal (95-100%): {np.sum(spo2_labels == 0)} ({np.sum(spo2_labels == 0)/len(spo2_labels)*100:.1f}%)")
    print(f"  Mild Hypoxemia (90-94%): {np.sum(spo2_labels == 1)} ({np.sum(spo2_labels == 1)/len(spo2_labels)*100:.1f}%)")
    print(f"  High Hypoxemia (<90%): {np.sum(spo2_labels == 2)} ({np.sum(spo2_labels == 2)/len(spo2_labels)*100:.1f}%)")

    print(f"HR Distribution:")
    print(f"  Normal (60-100 bpm): {np.sum(hr_labels == 0)} ({np.sum(hr_labels == 0)/len(hr_labels)*100:.1f}%)")
    print(f"  Bradycardia (<60 bpm): {np.sum(hr_labels == 1)} ({np.sum(hr_labels == 1)/len(hr_labels)*100:.1f}%)")
    print(f"  Tachycardia (>100 bpm): {np.sum(hr_labels == 2)} ({np.sum(hr_labels == 2)/len(hr_labels)*100:.1f}%)")

    return X_raw.astype(np.float32), spo2_labels.astype(np.int32), hr_labels.astype(np.int32)

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

def build_dual_output_model():
    """
    Build a model with 6 outputs: 3 for SpO2 classification + 3 for HR classification.
    This matches the Arduino code expectation where:
    - outputs 0-2: SpO2 classes (Normal, Mild Hypoxemia, High Hypoxemia)
    - outputs 3-5: HR classes (Normal, Bradycardia, Tachycardia)
    """
    inputs = layers.Input(shape=(2,), name='input')

    # Shared hidden layers
    x = layers.Dense(32, activation='relu', name='shared_dense1',
                    kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(24, activation='relu', name='shared_dense2',
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)

    # SpO2 branch
    spo2_branch = layers.Dense(16, activation='relu', name='spo2_dense',
                              kernel_regularizer=regularizers.l2(0.001))(x)
    spo2_branch = layers.Dropout(0.2)(spo2_branch)
    spo2_output = layers.Dense(3, activation='softmax', name='spo2_output')(spo2_branch)

    # HR branch
    hr_branch = layers.Dense(16, activation='relu', name='hr_dense',
                            kernel_regularizer=regularizers.l2(0.001))(x)
    hr_branch = layers.Dropout(0.2)(hr_branch)
    hr_output = layers.Dense(3, activation='softmax', name='hr_output')(hr_branch)

    # Concatenate outputs to match Arduino expectation (SpO2 first 3, HR next 3)
    combined_output = layers.Concatenate(name='combined_output')([spo2_output, hr_output])

    model = keras.Model(inputs=inputs, outputs=combined_output)

    # Custom loss function for multi-task learning
    def combined_loss(y_true, y_pred):
        # Extract SpO2 and HR parts
        spo2_true = y_true[:, 0]  # SpO2 labels
        hr_true = y_true[:, 1]    # HR labels

        spo2_pred = y_pred[:, 0:3]  # First 3 outputs
        hr_pred = y_pred[:, 3:6]    # Last 3 outputs

        # Calculate categorical crossentropy for each task
        spo2_loss = tf.keras.losses.sparse_categorical_crossentropy(spo2_true, spo2_pred)
        hr_loss = tf.keras.losses.sparse_categorical_crossentropy(hr_true, hr_pred)

        return spo2_loss + hr_loss

    def combined_accuracy(y_true, y_pred):
        spo2_true = y_true[:, 0]
        hr_true = y_true[:, 1]

        spo2_pred = tf.argmax(y_pred[:, 0:3], axis=1)
        hr_pred = tf.argmax(y_pred[:, 3:6], axis=1)

        spo2_acc = tf.cast(tf.equal(tf.cast(spo2_true, tf.int64), spo2_pred), tf.float32)
        hr_acc = tf.cast(tf.equal(tf.cast(hr_true, tf.int64), hr_pred), tf.float32)

        return (spo2_acc + hr_acc) / 2.0


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=combined_loss,
        metrics=[combined_accuracy]
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

def evaluate_dual_model(model, X_val, y_spo2_val, y_hr_val):
    """
    Evaluate the dual-output model for both SpO2 and HR classifications.
    """
    # Prepare validation labels for the model
    y_val_combined = np.column_stack([y_spo2_val, y_hr_val])

    # Get predictions
    y_pred = model.predict(X_val, verbose=0)

    # Split predictions
    spo2_pred_proba = y_pred[:, 0:3]
    hr_pred_proba = y_pred[:, 3:6]

    spo2_pred = np.argmax(spo2_pred_proba, axis=1)
    hr_pred = np.argmax(hr_pred_proba, axis=1)

    # Calculate accuracies
    spo2_acc = np.mean(spo2_pred == y_spo2_val)
    hr_acc = np.mean(hr_pred == y_hr_val)
    overall_acc = (spo2_acc + hr_acc) / 2.0

    print(f"\n📊 Detailed Evaluation Results:")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"SpO2 Classification Accuracy: {spo2_acc*100:.2f}%")
    print(f"HR Classification Accuracy: {hr_acc*100:.2f}%")

    # SpO2 Classification Report
    print("\n📋 SpO2 Classification Report:")
    spo2_target_names = ['Normal (95-100%)', 'Mild Hypoxemia (90-94%)', 'High Hypoxemia (<90%)']
    print(classification_report(y_spo2_val, spo2_pred, target_names=spo2_target_names))

    # HR Classification Report
    print("\n📋 HR Classification Report:")
    hr_target_names = ['Normal (60-100 bpm)', 'Bradycardia (<60 bpm)', 'Tachycardia (>100 bpm)']
    print(classification_report(y_hr_val, hr_pred, target_names=hr_target_names))

    return overall_acc

def main(args):
    print("🏥 Training Dual-Output Health Monitoring Model for Arduino")
    print("=" * 60)

    # 1) Generate realistic synthetic data
    print("🔄 Generating realistic synthetic data...")
    X_raw, y_spo2, y_hr = generate_realistic_synthetic_data(
        n_samples=args.samples,
        random_seed=42
    )

    # 2) Normalize features
    print("⚙️ Normalizing features...")
    X_norm = normalize_features(X_raw)

    # 3) Split data (train/val/test)
    X_temp, X_test, y_spo2_temp, y_spo2_test, y_hr_temp, y_hr_test = train_test_split(
        X_norm, y_spo2, y_hr, test_size=0.15, random_state=42, stratify=y_spo2
    )
    X_train, X_val, y_spo2_train, y_spo2_val, y_hr_train, y_hr_val = train_test_split(
        X_temp, y_spo2_temp, y_hr_temp, test_size=0.18, random_state=42, stratify=y_spo2_temp
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 4) Build and train model
    print("\n🏗️ Building dual-output model...")
    model = build_dual_output_model()
    model.summary()

    # Calculate model parameters
    total_params = model.count_params()
    print(f"📊 Total parameters: {total_params:,}")

    print("\n🚀 Training model...")

    # Prepare training data
    y_train_combined = np.column_stack([y_spo2_train, y_hr_train])
    y_val_combined = np.column_stack([y_spo2_val, y_hr_val])

    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_combined_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max' # Added mode='max'
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
        X_train, y_train_combined,
        validation_data=(X_val, y_val_combined),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 5) Evaluate model
    print("\n📈 Evaluating model performance...")
    val_accuracy = evaluate_dual_model(model, X_val, y_spo2_val, y_hr_val)

    # Test set evaluation
    y_test_combined = np.column_stack([y_spo2_test, y_hr_test])
    test_loss, test_acc = model.evaluate(X_test, y_test_combined, verbose=0)
    print(f"\n🎯 Final Test Accuracy: {test_acc*100:.2f}%")

    # 6) Convert to TFLite
    print("\n⚙️ Converting to TFLite for microcontroller...")
    tflite_model = convert_to_tflite_int8(model, X_train, args.tflite_path)

    if tflite_model:
        print("\n✅ Model training and conversion completed successfully!")
        print(f"📁 TFLite model saved as: {args.tflite_path}")
        print("🔧 Ready for deployment on Arduino!")
        print("\nModel output format:")
        print("  - Outputs 0-2: SpO2 classes (Normal, Mild Hypoxemia, High Hypoxemia)")
        print("  - Outputs 3-5: HR classes (Normal, Bradycardia, Tachycardia)")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train dual-output health monitoring model for Arduino"
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
        "--tflite_path", type=str, default="hrSpO2model.tflite",
        help="Output TFLite file path"
    )

    # Parse known arguments and ignore the rest
    args, unknown = parser.parse_known_args()

    model, history = main(args)
