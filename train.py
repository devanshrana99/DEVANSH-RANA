"""
Train & Evaluate the Traffic CNN Model
Run this script once to train the model before starting the server.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.cnn_model import TrafficCNNModel
from utils.data_generator import TrafficDataGenerator

def train_and_evaluate():
    print("=" * 60)
    print("  Real-Time Traffic Congestion Prediction — CNN Trainer")
    print("=" * 60)

    # 1. Generate data
    generator = TrafficDataGenerator()
    X_train, X_val, y_train, y_val = generator.get_train_val_split(n_samples=5000)

    print(f"\n📊 Data Shape:")
    print(f"   X_train: {X_train.shape}  (samples, time_steps, features, channels)")
    print(f"   X_val:   {X_val.shape}")
    print(f"   Classes: {np.bincount(y_train)} (Free/Moderate/Heavy/Severe)\n")

    # 2. Build model
    model = TrafficCNNModel(input_shape=(24, 10, 3), num_classes=4)
    model.summary()

    # 3. Train
    print("\n🔄 Training CNN Model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=40,
        batch_size=64
    )

    # 4. Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred, confidence, probs = model.predict(X_val)

    print("\n📈 Classification Report:")
    labels = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
    print(classification_report(y_val, y_pred, target_names=labels))

    acc = np.mean(y_pred == y_val)
    print(f"✅ Overall Accuracy: {acc * 100:.2f}%")
    print(f"✅ Avg Confidence:   {np.mean(confidence) * 100:.2f}%")

    # 5. Save model
    model.save("models/traffic_cnn_saved.h5")

    # 6. Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')
    for ax in axes:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#94a3b8')
        ax.spines['bottom'].set_color('#1e2d45')
        ax.spines['top'].set_color('#1e2d45')
        ax.spines['left'].set_color('#1e2d45')
        ax.spines['right'].set_color('#1e2d45')

    axes[0].plot(history.history['accuracy'], color='#00d4ff', label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#7c3aed', label='Validation')
    axes[0].set_title('Model Accuracy', color='white')
    axes[0].set_xlabel('Epoch', color='#94a3b8')
    axes[0].legend(facecolor='#1a2235', labelcolor='white')

    axes[1].plot(history.history['loss'], color='#f97316', label='Train')
    axes[1].plot(history.history['val_loss'], color='#ef4444', label='Validation')
    axes[1].set_title('Model Loss', color='white')
    axes[1].set_xlabel('Epoch', color='#94a3b8')
    axes[1].legend(facecolor='#1a2235', labelcolor='white')

    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=120, bbox_inches='tight',
                facecolor='#0a0e1a')
    print("\n📊 Training curves saved to models/training_curves.png")

    # 7. Demo prediction
    print("\n🚦 Demo Real-Time Prediction:")
    sample = generator.generate_realtime_sample()
    result = model.predict_single(sample)
    print(f"   Predicted Level:  {result['congestion_level']} — {result['congestion_label']}")
    print(f"   Confidence:       {result['confidence'] * 100:.1f}%")
    print(f"   Probabilities:")
    for lbl, prob in result['probabilities'].items():
        bar = '█' * int(prob * 20)
        print(f"     {lbl:12s}: {bar:<20s} {prob*100:.1f}%")

    print("\n✅ Training complete! Run app.py to start the server.")
    return model

if __name__ == '__main__':
    train_and_evaluate()
