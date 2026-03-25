import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import winsound

# ── SOUND ALERT SETUP ─────────────────────────────────────────────────────────
_sound_playing = False
_sound_lock    = threading.Lock()

def beep_alert(alert_type):
    global _sound_playing
    if _sound_playing:
        return
    def _run():
        global _sound_playing
        with _sound_lock:
            _sound_playing = True
            if alert_type == 'eyes':
                winsound.Beep(2000, 200)
                winsound.Beep(2000, 200)
                winsound.Beep(2000, 200)
            elif alert_type == 'yawn':
                winsound.Beep(1000, 600)
            elif alert_type == 'head':
                winsound.Beep(1500, 150)
                winsound.Beep(1500, 150)
                winsound.Beep(1500, 150)
            elif alert_type == 'danger':
                for _ in range(4):
                    winsound.Beep(2500, 200)
                    winsound.Beep(800,  200)
            _sound_playing = False
    threading.Thread(target=_run, daemon=True).start()


# ── MEDIAPIPE SETUP ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
EYE_THRESHOLD        = 0.18
YAWN_THRESHOLD       = 0.08
CLOSED_SECONDS_LIMIT = 2.0
PITCH_UP_LIMIT       = 0.25
PITCH_DOWN_LIMIT     = 0.58
YAW_LEFT_LIMIT       = 0.42
YAW_RIGHT_LIMIT      = 0.58

BUFFER_SIZE = 5
EAR_BUFFER  = []
YAWN_BUFFER = []

# ── LANDMARK INDICES ──────────────────────────────────────────────────────────
# Left eye outline
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eye outline
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Mouth outline
MOUTH     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             375, 321, 405, 314, 17, 84, 181, 91, 146]
# Face outline (for head box)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ── STATE ─────────────────────────────────────────────────────────────────────
carefulness_history = [100] * 50
closed_start_time   = None
eye_warning_count   = 0
yawn_warning_count  = 0
head_warning_count  = 0
is_yawning_latched  = False
was_drowsy_latched  = False
head_alert_latched  = False
prev_frame_time     = 0


# ── HELPERS ───────────────────────────────────────────────────────────────────
def smooth(buffer, new_val, size=BUFFER_SIZE):
    buffer.append(new_val)
    if len(buffer) > size:
        buffer.pop(0)
    return sum(buffer) / len(buffer)

def get_ear(landmarks, eye_indices, w, h):
    pts   = [np.array([landmarks.landmark[i].x * w,
                       landmarks.landmark[i].y * h]) for i in eye_indices]
    v1    = np.linalg.norm(pts[1] - pts[5])
    v2    = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * horiz)

def get_bounding_box(landmarks, indices, w, h, padding=4):
    """Get bounding box (x1,y1,x2,y2) around a group of landmarks."""
    xs = [int(landmarks.landmark[i].x * w) for i in indices]
    ys = [int(landmarks.landmark[i].y * h) for i in indices]
    return max(0, min(xs)-padding), max(0, min(ys)-padding), \
           min(w, max(xs)+padding), min(h, max(ys)+padding)

def get_drowsiness_level(score):
    if score >= 80:
        return "Alert",           (0, 255,   0)
    elif score >= 50:
        return "Slightly Drowsy", (0, 165, 255)
    else:
        return "Drowsy",          (0,   0, 255)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    curr_frame_time = time.time()
    fps = int(1 / (curr_frame_time - prev_frame_time)) if prev_frame_time > 0 else 0
    prev_frame_time = curr_frame_time

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results  = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_score = 100

    status   = "Alert"
    yawn_msg = ""
    tilt_msg = ""

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:

            # ── BOUNDING BOXES ────────────────────────────────────────────────

            # Head / face box (yellow)
            hx1, hy1, hx2, hy2 = get_bounding_box(landmarks, FACE_OVAL, w, h, padding=10)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
            cv2.putText(frame, "Head", (hx1, hy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Left eye box (green)
            lx1, ly1, lx2, ly2 = get_bounding_box(landmarks, LEFT_EYE, w, h)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)
            cv2.putText(frame, "L.Eye", (lx1, ly1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Right eye box (green)
            rx1, ry1, rx2, ry2 = get_bounding_box(landmarks, RIGHT_EYE, w, h)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
            cv2.putText(frame, "R.Eye", (rx1, ry1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Mouth box (orange — turns red when yawning)
            mx1, my1, mx2, my2 = get_bounding_box(landmarks, MOUTH, w, h)
            mouth_color = (0, 0, 255) if yawn_msg else (0, 165, 255)
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), mouth_color, 1)
            cv2.putText(frame, "Mouth", (mx1, my1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, mouth_color, 1)

            # ── A. CALCULATIONS ───────────────────────────────────────────────
            raw_ear = (get_ear(landmarks, [362, 385, 387, 263, 373, 380], w, h) +
                       get_ear(landmarks, [33,  160, 158, 133, 153, 144], w, h)) / 2.0
            avg_ear = smooth(EAR_BUFFER, raw_ear)

            raw_lip  = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
            lip_dist = smooth(YAWN_BUFFER, raw_lip)

            pitch = abs(landmarks.landmark[1].y  - landmarks.landmark[10].y) / \
                    abs(landmarks.landmark[152].y - landmarks.landmark[10].y)

            nose_x     = landmarks.landmark[1].x
            left_face  = landmarks.landmark[234].x
            right_face = landmarks.landmark[454].x
            face_width = right_face - left_face if right_face > left_face else 0.001
            nose_ratio = (nose_x - left_face) / face_width

            # ── B. ALERT LOGIC ────────────────────────────────────────────────

            # 1. Eye Closure — eye boxes turn red when closing
            if avg_ear < EYE_THRESHOLD:
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                if closed_start_time is None:
                    closed_start_time = time.time()
                duration = time.time() - closed_start_time
                if duration >= CLOSED_SECONDS_LIMIT:
                    status = "WATCH OUT! OPEN YOUR EYES"
                    current_score -= 40
                    if not was_drowsy_latched:
                        eye_warning_count += 1
                        beep_alert('eyes')
                        was_drowsy_latched = True
                else:
                    status = "Eyes Closing..."
                    current_score -= 10
            else:
                closed_start_time  = None
                was_drowsy_latched = False

            # 2. Yawn — mouth box turns red when yawning
            if lip_dist > YAWN_THRESHOLD:
                yawn_msg = "YOU ARE TIRED, PULL OVER"
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
                current_score -= 30
                if not is_yawning_latched:
                    yawn_warning_count += 1
                    beep_alert('yawn')
                    is_yawning_latched = True
            else:
                is_yawning_latched = False

            # 3. Head Position — head box turns red when off position
            tilt_msgs = []
            if pitch < PITCH_UP_LIMIT:
                tilt_msgs.append("LOOKING UP")
            elif pitch > PITCH_DOWN_LIMIT:
                tilt_msgs.append("NODDING DOWN")
            if nose_ratio < YAW_LEFT_LIMIT:
                tilt_msgs.append("LOOKING LEFT")
            elif nose_ratio > YAW_RIGHT_LIMIT:
                tilt_msgs.append("LOOKING RIGHT")

            if tilt_msgs:
                tilt_msg = " + ".join(tilt_msgs)
                current_score -= 20 * len(tilt_msgs)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                if not head_alert_latched:
                    head_warning_count += 1
                    beep_alert('head')
                    head_alert_latched = True
            else:
                head_alert_latched = False

            # Emergency
            if current_score < 50 and not _sound_playing:
                beep_alert('danger')

            # ── C. DROWSINESS LEVEL ───────────────────────────────────────────
            carefulness_history.append(max(0, current_score))
            carefulness_history.pop(0)
            level_text, level_color = get_drowsiness_level(current_score)

            # ── D. UI PANEL ───────────────────────────────────────────────────
            cv2.rectangle(frame, (10, 10), (460, 235), (0, 0, 0), -1)

            status_color = (0,0,255) if current_score < 50 else \
                           (0,165,255) if current_score < 80 else (0,255,0)

            cv2.putText(frame, f"STATUS : {status}",
                        (20, 42),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(frame, f"LEVEL  : {level_text}",
                        (20, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, level_color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.3f}  PITCH: {pitch:.3f}  YAW: {nose_ratio:.2f}",
                        (20, 96),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
            cv2.putText(frame, f"EYE WARNINGS : {eye_warning_count}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"YAWN WARNINGS: {yawn_warning_count}",
                        (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"HEAD WARNINGS: {head_warning_count}",
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            if yawn_msg:
                cv2.putText(frame, yawn_msg,
                            (20, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)
            if tilt_msg:
                cv2.putText(frame, f"HEAD: {tilt_msg}",
                            (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,255), 2)

            # ── E. ALERTNESS GRAPH ────────────────────────────────────────────
            gx, gy = w - 220, h - 30
            cv2.rectangle(frame, (gx, gy-100), (gx+200, gy), (50,50,50), -1)
            for i in range(1, len(carefulness_history)):
                pt1 = (gx+(i-1)*4, gy - carefulness_history[i-1])
                pt2 = (gx+i*4,     gy - carefulness_history[i])
                cv2.line(frame, pt1, pt2, (0,255,0), 2)
            cv2.putText(frame, "Alertness", (gx, gy-105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    else:
        cv2.putText(frame, "No face detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(frame, f"FPS: {fps}", (w-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow('DriveSafe AI - Competition Final', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()