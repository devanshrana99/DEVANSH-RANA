"""
CNN Model for Real-Time Traffic Congestion Prediction
Uses Convolutional Neural Networks to analyze traffic patterns
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os


class TrafficCNNModel:
    """
    CNN-based Traffic Congestion Prediction Model
    
    Architecture:
    - Input: Traffic feature maps (speed, volume, occupancy over time)
    - Conv layers: Extract spatial-temporal patterns
    - Dense layers: Classify congestion level
    - Output: Congestion probability & level (0=Free, 1=Moderate, 2=Heavy, 3=Severe)
    """

    CONGESTION_LEVELS = {
        0: "Free Flow",
        1: "Moderate",
        2: "Heavy",
        3: "Severe"
    }

    CONGESTION_COLORS = {
        0: "#00C851",   # Green
        1: "#ffbb33",   # Yellow
        2: "#FF8800",   # Orange
        3: "#ff4444"    # Red
    }

    def __init__(self, input_shape=(24, 10, 3), num_classes=4):
        """
        Args:
            input_shape: (time_steps, num_features, channels)
                         24 time steps (hourly), 10 road features, 3 channels
            num_classes: Number of congestion levels
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """Build the CNN architecture"""
        inputs = keras.Input(shape=self.input_shape, name="traffic_input")

        # --- Block 1: Local Pattern Extraction ---
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.25)(x)

        # --- Block 2: Mid-level Feature Extraction ---
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.25)(x)

        # --- Block 3: High-level Pattern Recognition ---
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (1, 1), padding='same', activation='relu', name='conv3_2')(x)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.Dropout(0.4)(x)

        # --- Dense Classification Head ---
        x = layers.Dense(256, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.3)(x)

        # Output: probabilities for each congestion class
        outputs = layers.Dense(self.num_classes, activation='softmax', name='congestion_output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='TrafficCNN')
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the CNN model"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X):
        """Predict congestion level"""
        probs = self.model.predict(X, verbose=0)
        class_idx = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        return class_idx, confidence, probs

    def predict_single(self, features: np.ndarray):
        """Predict for a single sample"""
        X = features.reshape(1, *self.input_shape)
        class_idx, confidence, probs = self.predict(X)
        level = int(class_idx[0])
        return {
            "congestion_level": level,
            "congestion_label": self.CONGESTION_LEVELS[level],
            "congestion_color": self.CONGESTION_COLORS[level],
            "confidence": float(confidence[0]),
            "probabilities": {
                self.CONGESTION_LEVELS[i]: float(probs[0][i])
                for i in range(self.num_classes)
            }
        }

    def save(self, path="models/traffic_cnn.h5"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path="models/traffic_cnn.h5"):
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def summary(self):
        return self.model.summary()
