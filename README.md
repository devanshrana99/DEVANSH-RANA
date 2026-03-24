# 🚦 Real-Time Traffic Congestion Prediction System
 
A fully functional traffic prediction system using **Python**, **CNN (TensorFlow/Keras)**, and **AIML** chatbot.
 
---
 
## 📁 Project Structure
 
```
traffic_prediction/
├── app.py                    # Flask REST API server
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
│
├── models/
│   └── cnn_model.py          # CNN architecture (TensorFlow/Keras)
│
├── utils/
│   ├── data_generator.py     # Synthetic traffic data generator
│   ├── chatbot.py            # AIML chatbot engine
│   └── traffic_bot.aiml      # AIML knowledge base
│
└── templates/
    └── dashboard.html        # Live dashboard UI
```
 
---
 
## 🧠 CNN Architecture
 
```
Input (24, 10, 3)          ← 24 hours, 10 features, 3 channels
    ↓
Conv2D(32) × 2 + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D(64) × 2 + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2D(128) × 2 + BatchNorm + GlobalAvgPool + Dropout(0.4)
    ↓
Dense(256) → Dense(128) → Dense(4, Softmax)
    ↓
Output: [Free Flow, Moderate, Heavy, Severe]
```
 
**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Dataset: 5000 synthetic samples (~91% accuracy)
 
---
 
## 📡 10 Traffic Features
 
| Feature | Description |
|---------|-------------|
| avg_speed | Average vehicle speed (normalized, 0-120 km/h) |
| volume | Vehicles per minute |
| occupancy | % road occupied |
| incidents | Accident/incident count |
| weather_score | Weather impact (0=bad, 1=clear) |
| time_sin | Time-of-day (sine encoding) |
| time_cos | Time-of-day (cosine encoding) |
| capacity_util | Road capacity utilization |
| temperature | Ambient temperature (normalized) |
| visibility | Visibility (normalized) |
 
---
 
## 🤖 AIML Chatbot Topics
 
The TrafficBot handles natural language queries:
- Traffic conditions & congestion levels
- How the CNN model works
- Best travel times
- Route recommendations
- Weather impact on traffic
- Road segment info (NH-48, Ring Road, etc.)
 
---
 
## 🚀 Quick Start
 
### 1. Install Dependencies
```bash
cd traffic_prediction
pip install -r requirements.txt
```
 
### 2. Train the CNN Model (first time only)
```bash
python train.py
```
This generates `models/traffic_cnn_saved.h5` and saves training curves.
 
### 3. Start the Server
```bash
python app.py
```
 
### 4. Open Dashboard
```
http://localhost:5000
```
 
---
 
## 🔌 REST API Endpoints
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Live dashboard |
| GET | `/api/predict/all` | Predictions for all segments |
| GET | `/api/predict/<id>` | Prediction for one segment |
| POST | `/api/chat` | AIML chatbot |
| GET | `/api/history?segment=NH48_Delhi` | 24h history |
| GET | `/api/model/info` | CNN architecture info |
| GET | `/api/segments` | All road segments |
| GET | `/api/stats` | System stats |
 
### Example API Call
```python
import requests
 
# Get all predictions
r = requests.get('http://localhost:5000/api/predict/all')
data = r.json()
for seg in data['predictions']:
    print(f"{seg['name']}: {seg['congestion_label']} ({seg['confidence']*100:.1f}%)")
 
# Chat with the bot
r = requests.post('http://localhost:5000/api/chat',
                  json={'message': 'What is the best time to travel?'})
print(r.json()['response'])
```
 
---
 
## 🗺️ Monitored Road Segments (Delhi)
 
1. **NH-48** — Delhi-Gurgaon Highway
2. **Ring Road** — Delhi Inner Ring Road
3. **Delhi-Meerut Expressway** — NH-9 North
4. **Airport Metro Express Road** — IGI Airport access
5. **Outer Ring Road** — Delhi bypass
6. **NH-9 Noida** — Delhi-Noida corridor
 
---
 
## 📊 Congestion Levels
 
| Level | Label | Speed | Color |
|-------|-------|-------|-------|
| 0 | Free Flow | > 90 km/h | 🟢 Green |
| 1 | Moderate | 60-90 km/h | 🟡 Yellow |
| 2 | Heavy | 30-60 km/h | 🟠 Orange |
| 3 | Severe | < 30 km/h | 🔴 Red |
 
---
 
## 🔧 Tech Stack
 
| Component | Technology |
|-----------|------------|
| ML Model | CNN — TensorFlow/Keras |
| Chatbot | AIML 1.0 (custom Python engine) |
| Backend | Python Flask REST API |
| Frontend | HTML/CSS/JS + Chart.js |
| Data | Synthetic sensor simulation |
| Updates | Background thread (60s refresh) |
 
---
 
## 📈 Extending the Project
 
- **Real data**: Connect to HERE Maps, TomTom, or OpenTraffic APIs
- **LSTM**: Add LSTM layers for better temporal modeling
- **Alerts**: Add email/SMS alerts for severe congestion
- **Mobile**: Build React Native app using the REST API
- **Database**: Add PostgreSQL for historical data storage
