"""
Traffic Data Generator & Preprocessor
Simulates real-world traffic sensor data for training and real-time prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import random


class TrafficDataGenerator:
    """
    Generates realistic synthetic traffic data for training the CNN model.
    
    Features generated:
    - avg_speed: Average vehicle speed (km/h)
    - volume: Vehicles per minute
    - occupancy: % of time road is occupied
    - incidents: Number of incidents/accidents
    - weather_score: Weather impact score (0-1)
    - time_of_day: Hour of day (normalized)
    - day_of_week: Day (0=Mon, 6=Sun)
    - road_capacity: Max capacity utilization
    - temp_celsius: Temperature
    - visibility: Visibility in meters (normalized)
    """

    ROAD_SEGMENTS = [
        {"id": "NH48_Delhi", "name": "NH-48 Delhi-Gurgaon", "lat": 28.5355, "lng": 77.3910, "capacity": 3000},
        {"id": "Ring_Road", "name": "Ring Road Delhi", "lat": 28.6139, "lng": 77.2090, "capacity": 2500},
        {"id": "Expressway_N", "name": "Delhi-Meerut Expressway", "lat": 28.6692, "lng": 77.4538, "capacity": 2800},
        {"id": "Airport_Rd", "name": "Airport Metro Express Rd", "lat": 28.5562, "lng": 77.1000, "capacity": 2200},
        {"id": "Outer_Ring", "name": "Outer Ring Road", "lat": 28.6300, "lng": 77.3800, "capacity": 3200},
        {"id": "NH9_Noida", "name": "NH-9 Delhi-Noida", "lat": 28.5706, "lng": 77.3219, "capacity": 2600},
    ]

    def __init__(self, time_steps=24, n_features=10, seed=42):
        self.time_steps = time_steps
        self.n_features = n_features
        self.n_channels = 3  # Speed, Volume, Composite
        self.scaler = MinMaxScaler()
        np.random.seed(seed)
        random.seed(seed)

    def _congestion_from_hour(self, hour, day_of_week):
        """Determine base congestion level based on time patterns"""
        # Morning rush: 8-10 AM
        if 8 <= hour <= 10:
            base = 0.8 if day_of_week < 5 else 0.4
        # Evening rush: 5-8 PM
        elif 17 <= hour <= 20:
            base = 0.85 if day_of_week < 5 else 0.5
        # Lunch: 12-2 PM
        elif 12 <= hour <= 14:
            base = 0.55 if day_of_week < 5 else 0.3
        # Night: 11 PM - 5 AM
        elif hour >= 23 or hour <= 5:
            base = 0.1
        # Weekend baseline
        elif day_of_week >= 5:
            base = 0.25
        else:
            base = 0.35
        return base + np.random.normal(0, 0.08)

    def _generate_features(self, hour, day_of_week, weather_factor=1.0):
        """Generate a feature vector for a given time/condition"""
        congestion_ratio = np.clip(self._congestion_from_hour(hour, day_of_week), 0, 1)

        # Speed inversely related to congestion
        max_speed = 120
        avg_speed = max_speed * (1 - congestion_ratio * 0.85) + np.random.normal(0, 5)
        avg_speed = np.clip(avg_speed, 5, max_speed)

        # Volume positively related to congestion
        max_volume = 50
        volume = max_volume * congestion_ratio * weather_factor + np.random.normal(0, 3)
        volume = np.clip(volume, 0, max_volume)

        # Occupancy
        occupancy = congestion_ratio * 0.95 + np.random.normal(0, 0.05)
        occupancy = np.clip(occupancy, 0, 1)

        # Incidents (more during heavy congestion)
        incidents = int(np.random.poisson(congestion_ratio * 2))

        # Weather
        weather_score = weather_factor + np.random.normal(0, 0.05)
        weather_score = np.clip(weather_score, 0, 1)

        # Time features
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)

        # Other features
        capacity_util = congestion_ratio * 1.1 + np.random.normal(0, 0.05)
        temp = 25 + np.random.normal(0, 10)
        visibility = (1 - 0.3 * (1 - weather_factor)) + np.random.normal(0, 0.05)

        features = [
            avg_speed / 120,        # Normalized speed
            volume / 50,            # Normalized volume
            occupancy,              # Occupancy ratio
            incidents / 10,         # Normalized incidents
            weather_score,          # Weather impact
            time_sin,               # Time (sine)
            time_cos,               # Time (cosine)
            np.clip(capacity_util, 0, 1),  # Capacity utilization
            np.clip((temp + 20) / 80, 0, 1),  # Normalized temperature
            np.clip(visibility, 0, 1)   # Visibility
        ]
        return np.array(features), congestion_ratio

    def _congestion_label(self, ratio):
        """Convert ratio to 4-class label"""
        if ratio < 0.25:
            return 0  # Free Flow
        elif ratio < 0.55:
            return 1  # Moderate
        elif ratio < 0.78:
            return 2  # Heavy
        else:
            return 3  # Severe

    def generate_training_data(self, n_samples=5000):
        """Generate (X, y) pairs for CNN training"""
        X, y = [], []

        for _ in range(n_samples):
            day_of_week = random.randint(0, 6)
            start_hour = random.randint(0, 23)
            weather = random.uniform(0.5, 1.0)

            # Build time window of 24 hours
            feature_sequence = []
            labels_in_window = []
            for t in range(self.time_steps):
                hour = (start_hour + t) % 24
                feats, ratio = self._generate_features(hour, day_of_week, weather)
                feature_sequence.append(feats)
                labels_in_window.append(ratio)

            # Shape: (24, 10) -> (24, 10, 3) channels
            seq = np.array(feature_sequence)  # (24, 10)
            speed_channel = seq[:, :1] * np.ones((1, self.n_features))
            volume_channel = seq[:, 1:2] * np.ones((1, self.n_features))
            composite = seq

            # Stack into 3 channels: (24, 10, 3)
            sample = np.stack([seq, seq * 0.9 + np.random.normal(0, 0.01, seq.shape),
                               seq * 1.1 - np.random.normal(0, 0.01, seq.shape)], axis=-1)

            # Label: congestion at last time step (prediction target)
            label = self._congestion_label(labels_in_window[-1])

            X.append(sample)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def get_train_val_split(self, n_samples=5000, val_ratio=0.2):
        """Generate and split data"""
        print(f"Generating {n_samples} traffic samples...")
        X, y = self.generate_training_data(n_samples)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        return X_train, X_val, y_train, y_val

    def generate_realtime_sample(self, hour=None, day_of_week=None, weather=1.0):
        """Generate a real-time sample for live prediction"""
        if hour is None:
            hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()

        feature_sequence = []
        for t in range(self.time_steps):
            h = (hour - self.time_steps + t + 1) % 24
            feats, _ = self._generate_features(h, day_of_week, weather)
            feature_sequence.append(feats)

        seq = np.array(feature_sequence)
        sample = np.stack([
            seq,
            seq * 0.9 + np.random.normal(0, 0.01, seq.shape),
            seq * 1.1 - np.random.normal(0, 0.01, seq.shape)
        ], axis=-1)
        return sample.astype(np.float32)

    def get_all_segments_realtime(self, weather=1.0):
        """Generate real-time data for all road segments"""
        now = datetime.now()
        hour = now.hour
        dow = now.weekday()
        results = []

        for seg in self.ROAD_SEGMENTS:
            # Slight variation per segment
            seg_weather = weather * random.uniform(0.85, 1.0)
            sample = self.generate_realtime_sample(hour, dow, seg_weather)
            results.append({
                "segment": seg,
                "sample": sample,
                "hour": hour,
                "day_of_week": dow
            })
        return results
