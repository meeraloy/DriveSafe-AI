import cv2
import mediapipe as mp
import time
import numpy as np
import winsound

# --- 1. SETUP MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. SETTINGS ---
EYE_THRESHOLD = 0.22
YAWN_THRESHOLD = 0.06
CLOSED_SECONDS_LIMIT = 2.0
PITCH_UP_LIMIT = 0.30
PITCH_DOWN_LIMIT = 0.52

# Tracking
carefulness_history = [100] * 50
closed_start_time = None
eye_warning_count = 0
yawn_warning_count = 0
is_yawning_latched = False
was_drowsy_latched = False

# FPS Tracking Variables (NEW)
prev_frame_time = 0


# --- 3. HELPERS ---
def get_ear(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * horiz)


# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    # Calculate FPS (NEW)
    curr_frame_time = time.time()
    fps = int(1 / (curr_frame_time - prev_frame_time)) if prev_frame_time > 0 else 0
    prev_frame_time = curr_frame_time

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_score = 100

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # A. CALCULATIONS
            avg_ear = (get_ear(landmarks, [362, 385, 387, 263, 373, 380], w, h) +
                       get_ear(landmarks, [33, 160, 158, 133, 153, 144], w, h)) / 2.0
            lip_dist = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
            pitch = abs(landmarks.landmark[1].y - landmarks.landmark[10].y) / \
                    abs(landmarks.landmark[152].y - landmarks.landmark[10].y)

            # --- B. ALERT LOGIC ---
            status = "Alert"

            # 1. Eye Closure Logic (BEEP BEEP)
            if avg_ear < EYE_THRESHOLD:
                if closed_start_time is None: closed_start_time = time.time()
                duration = time.time() - closed_start_time
                if duration >= CLOSED_SECONDS_LIMIT:
                    status = "WATCH OUT! OPEN YOUR EYES"
                    current_score -= 40
                    if not was_drowsy_latched:
                        eye_warning_count += 1
                        # Double beep for eyes
                        winsound.Beep(2000, 100)
                        winsound.Beep(2000, 100)
                        was_drowsy_latched = True
                else:
                    status = "Eyes Closed..."
            else:
                closed_start_time = None
                was_drowsy_latched = False

            # 2. Yawn Logic (BEEP)
            yawn_msg = ""
            if lip_dist > YAWN_THRESHOLD:
                yawn_msg = "YOU ARE TIRED, PULL OVER"
                current_score -= 30
                if not is_yawning_latched:
                    yawn_warning_count += 1
                    # Single long beep for yawn
                    winsound.Beep(1000, 400)
                    is_yawning_latched = True
            else:
                is_yawning_latched = False

            # 3. Head Position (Silent)
            tilt_msg = ""
            if pitch < PITCH_UP_LIMIT:
                tilt_msg = "LOOKING UP"
                current_score -= 20
            elif pitch > PITCH_DOWN_LIMIT:
                tilt_msg = "NODDING DOWN"
                current_score -= 20

            # --- C. UI & GRAPH ---
            carefulness_history.append(max(0, current_score))
            carefulness_history.pop(0)

            # Dash Panel
            cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
            color = (0, 0, 255) if current_score < 70 else (0, 255, 0)
            cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"EYE WARNINGS: {eye_warning_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
            cv2.putText(frame, f"YAWN WARNINGS: {yawn_warning_count}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
            if yawn_msg: cv2.putText(frame, yawn_msg, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if tilt_msg: cv2.putText(frame, f"POS: {tilt_msg}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255),
                                     2)

            # Graph Drawing
            graph_x, graph_y = w - 220, h - 30
            cv2.rectangle(frame, (graph_x, graph_y - 100), (graph_x + 200, graph_y), (50, 50, 50), -1)
            for i in range(1, len(carefulness_history)):
                cv2.line(frame, (graph_x + (i - 1) * 4, graph_y - carefulness_history[i - 1]),
                         (graph_x + i * 4, graph_y - carefulness_history[i]), (0, 255, 0), 2)

    # Display FPS in top right corner (NEW)
    cv2.putText(frame, f"FPS: {fps}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('DriveSafe AI - Competition Final', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()