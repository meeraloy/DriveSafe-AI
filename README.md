# 🚗 DriveSafe AI — Real-Time Driver Drowsiness Detection

> **An AI-powered system that watches for signs of driver fatigue in real time — and alerts before it's too late.**

---

## 🧠 What is it?

DriveSafe AI uses your webcam to detect drowsiness while driving. It tracks your **eyes**, **yawns**, and **head position** every single frame — and fires an audio + visual alert the moment something looks off.

No internet needed. Runs fully on your machine. Streams live to a browser dashboard.

---

## 🎯 The Problem

Drowsy driving causes thousands of accidents every year. Most drivers don't notice fatigue setting in until it's already affecting them. DriveSafe AI acts as an intelligent co-pilot that **never gets tired**.

---

## ⚙️ How It Works

### 👁️ 1. Eye Closure Detection
- Uses **MediaPipe's 468 facial landmarks** to calculate the **Eye Aspect Ratio (EAR)**
- If EAR < `0.18` for **2+ seconds** → alert triggered
- Tracks left and right eyes independently

### 😮 2. Yawn Detection
- Measures the vertical distance between upper and lower lip landmarks
- Threshold: `0.045` — calibrated for natural, relaxed yawns
- Each yawn counted once (latch system prevents rapid re-triggering)

### 🙆 3. Head Pose Detection
- Calculates **pitch** (nodding) and **yaw** (turning) from nose/chin/forehead landmarks
- Alerts: `NODDING DOWN`, `LOOKING UP`, `LOOKING LEFT`, `LOOKING RIGHT`

### 🧬 4. CNN Eye Model
- Custom **Convolutional Neural Network** trained on eye images
- Runs on **TensorFlow Lite** for real-time speed
- Acts as a secondary verification layer alongside EAR

---

## 📊 Alertness Score

Every frame calculates a score from **0–100**:

| Event | Penalty | Level |
|---|---|---|
| Eyes closing | -10 | Slightly Drowsy |
| Eyes closed 2s+ | -40 | Drowsy / Danger |
| Yawning | -30 | Drowsy |
| Head tilt/turn | -20 per direction | Warning |

---

## 🏆 Results

| Metric | Result |
|---|---|
| Training Accuracy | ~99% |
| Validation Accuracy | ~99% |
| Overfitting | None |
| Real-time FPS | ~25–30 FPS |

Both curves converged cleanly at epoch 14 with no overfitting detected.

---

## 🌐 Live Web Dashboard

The system streams to a browser at **`http://localhost:5000`** via Flask + MJPEG:

- 📹 Live webcam feed with landmark overlays
- 📈 Real-time alertness score & live graph
- ⚠️ Eye / Yawn / Head warning counters
- ⏱️ Session timer

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core logic |
| MediaPipe | 468 facial landmark detection |
| OpenCV | Webcam capture & frame rendering |
| TensorFlow Lite | CNN eye model inference |
| Flask | Web dashboard & video streaming |
| Winsound | Audio alerts (Windows) |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install opencv-python mediapipe numpy flask tensorflow

# 2. Run the app
python app.py

# 3. Open your browser
http://localhost:5000
```

> ⚠️ **Windows only** — audio alerts use `winsound`

---

## 📁 Project Structure

```
DriveSafe-AI/
├── app.py                  # Flask app + detection logic
├── main.py                 # Standalone (non-web) version
├── dashboard.html          # Web dashboard UI
├── eye_cnn_model.h5        # Trained CNN model
├── train_eye_cnn.py        # CNN training script
├── training_curves.png     # Accuracy & loss graphs
└── dataset/                # Eye image dataset
```

---

## 📌 Key Settings

```python
EYE_THRESHOLD        = 0.18    # Below this = eyes closed
YAWN_THRESHOLD       = 0.045   # Lip distance for yawn
CLOSED_SECONDS_LIMIT = 2.0     # Seconds before eye alert
```

---

*Built with ❤️ using Python, MediaPipe, and OpenCV*
