from flask import Flask, Response, render_template_string, jsonify
import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import winsound
import os
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
# ... rest of your imports
app = Flask(__name__)

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
                winsound.Beep(2000, 200); winsound.Beep(2000, 200); winsound.Beep(2000, 200)
            elif alert_type == 'yawn':
                winsound.Beep(1000, 600)
            elif alert_type == 'head':
                winsound.Beep(1500, 150); winsound.Beep(1500, 150); winsound.Beep(1500, 150)
            elif alert_type == 'danger':
                for _ in range(4):
                    winsound.Beep(2500, 200); winsound.Beep(800, 200)
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
YAWN_THRESHOLD       = 0.045   # reduced threshold
CLOSED_SECONDS_LIMIT = 2.0
PITCH_UP_LIMIT       = 0.25
PITCH_DOWN_LIMIT     = 0.58
YAW_LEFT_LIMIT       = 0.42
YAW_RIGHT_LIMIT      = 0.58
BUFFER_SIZE          = 5

LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ── SHARED STATE (thread-safe reads for the /stats endpoint) ──────────────────
state = {
    "status":      "Alert",
    "level":       "Alert",
    "score":       100,
    "ear":         0.0,
    "pitch":       0.0,
    "yaw":         0.0,
    "lip":         0.0,
    "eye_warns":   0,
    "yawn_warns":  0,
    "head_warns":  0,
    "yawn_msg":    "",
    "tilt_msg":    "",
    "fps":         0,
    "session_secs": 0,
}
state_lock = threading.Lock()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def smooth(buf, val, size=BUFFER_SIZE):
    buf.append(val)
    if len(buf) > size: buf.pop(0)
    return sum(buf) / len(buf)

def get_ear(landmarks, idx, w, h):
    pts = [np.array([landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]) for i in idx]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    return (v1 + v2) / (2.0 * np.linalg.norm(pts[0] - pts[3]))

def get_bbox(landmarks, indices, w, h, pad=4):
    xs = [int(landmarks.landmark[i].x * w) for i in indices]
    ys = [int(landmarks.landmark[i].y * h) for i in indices]
    return max(0, min(xs)-pad), max(0, min(ys)-pad), min(w, max(xs)+pad), min(h, max(ys)+pad)

def draw_label_box(frame, x1, y1, x2, y2, color, label, thickness=1):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

# ── CAMERA THREAD ─────────────────────────────────────────────────────────────
output_frame = None
frame_lock   = threading.Lock()

def camera_loop():
    global output_frame, _sound_playing

    ear_buf, yawn_buf           = [], []
    carefulness_history         = [100] * 50
    closed_start_time           = None
    eye_warning_count           = 0
    yawn_warning_count          = 0
    head_warning_count          = 0
    is_yawning_latched          = False
    was_drowsy_latched          = False
    head_alert_latched          = False
    prev_frame_time             = 0
    session_start               = time.time()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        curr = time.time()
        fps  = int(1 / (curr - prev_frame_time)) if prev_frame_time > 0 else 0
        prev_frame_time = curr

        frame    = cv2.flip(frame, 1)
        h, w, _  = frame.shape
        results  = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        current_score = 100
        status   = "Alert"
        yawn_msg = ""
        tilt_msg = ""

        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:

                # Bounding boxes
                hx1,hy1,hx2,hy2 = get_bbox(lm, FACE_OVAL, w, h, 10)
                lx1,ly1,lx2,ly2 = get_bbox(lm, LEFT_EYE,  w, h)
                rx1,ry1,rx2,ry2 = get_bbox(lm, RIGHT_EYE, w, h)
                mx1,my1,mx2,my2 = get_bbox(lm, MOUTH,     w, h)

                draw_label_box(frame, hx1,hy1,hx2,hy2, (0,255,255), "Head")
                draw_label_box(frame, lx1,ly1,lx2,ly2, (0,255,0),   "L.Eye")
                draw_label_box(frame, rx1,ry1,rx2,ry2, (0,255,0),   "R.Eye")
                draw_label_box(frame, mx1,my1,mx2,my2, (0,165,255), "Mouth")

                # Calculations
                raw_ear = (get_ear(lm, [362,385,387,263,373,380], w, h) +
                           get_ear(lm, [33, 160,158,133,153,144], w, h)) / 2.0
                avg_ear  = smooth(ear_buf, raw_ear)
                raw_lip  = abs(lm.landmark[13].y - lm.landmark[14].y)
                lip_dist = smooth(yawn_buf, raw_lip)
                pitch    = abs(lm.landmark[1].y  - lm.landmark[10].y) / \
                           abs(lm.landmark[152].y - lm.landmark[10].y)
                nose_x     = lm.landmark[1].x
                left_face  = lm.landmark[234].x
                right_face = lm.landmark[454].x
                fw         = right_face - left_face if right_face > left_face else 0.001
                nose_ratio = (nose_x - left_face) / fw

                # 1. Eye closure
                if avg_ear < EYE_THRESHOLD:
                    cv2.rectangle(frame,(lx1,ly1),(lx2,ly2),(0,0,255),2)
                    cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(0,0,255),2)
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

                # 2. Yawn
                if lip_dist > YAWN_THRESHOLD:
                    yawn_msg = "YOU ARE TIRED, PULL OVER"
                    cv2.rectangle(frame,(mx1,my1),(mx2,my2),(0,0,255),2)
                    current_score -= 30
                    if not is_yawning_latched:
                        yawn_warning_count += 1
                        beep_alert('yawn')
                        is_yawning_latched = True
                else:
                    is_yawning_latched = False

                # 3. Head pose
                tilt_msgs = []
                if pitch < PITCH_UP_LIMIT:        tilt_msgs.append("LOOKING UP")
                elif pitch > PITCH_DOWN_LIMIT:    tilt_msgs.append("NODDING DOWN")
                if nose_ratio < YAW_LEFT_LIMIT:   tilt_msgs.append("LOOKING LEFT")
                elif nose_ratio > YAW_RIGHT_LIMIT: tilt_msgs.append("LOOKING RIGHT")

                if tilt_msgs:
                    tilt_msg = " + ".join(tilt_msgs)
                    current_score -= 20 * len(tilt_msgs)
                    cv2.rectangle(frame,(hx1,hy1),(hx2,hy2),(0,0,255),2)
                    if not head_alert_latched:
                        head_warning_count += 1
                        beep_alert('head')
                        head_alert_latched = True
                else:
                    head_alert_latched = False

                if current_score < 50 and not _sound_playing:
                    beep_alert('danger')

                # Score & level
                carefulness_history.append(max(0, current_score))
                carefulness_history.pop(0)

                if current_score >= 80:   level = "Alert"
                elif current_score >= 50: level = "Slightly Drowsy"
                else:                     level = "Drowsy"

                # UI panel on frame
                cv2.rectangle(frame, (10, 10), (460, 235), (0,0,0), -1)
                sc = (0,0,255) if current_score < 50 else (0,165,255) if current_score < 80 else (0,255,0)
                lc = (0,0,255) if level=="Drowsy" else (0,165,255) if level=="Slightly Drowsy" else (0,255,0)
                cv2.putText(frame, f"STATUS : {status}",          (20,42),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)
                cv2.putText(frame, f"LEVEL  : {level}",           (20,70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, lc, 2)
                cv2.putText(frame, f"EAR:{avg_ear:.3f} PITCH:{pitch:.3f} YAW:{nose_ratio:.2f}", (20,96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
                cv2.putText(frame, f"EYE WARNINGS : {eye_warning_count}",  (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(frame, f"YAWN WARNINGS: {yawn_warning_count}", (20,145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(frame, f"HEAD WARNINGS: {head_warning_count}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                if yawn_msg:
                    cv2.putText(frame, yawn_msg, (20,198), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)
                if tilt_msg:
                    cv2.putText(frame, f"HEAD: {tilt_msg}", (20,225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,255), 2)

                # Alertness graph
                gx, gy = w-220, h-30
                cv2.rectangle(frame,(gx,gy-100),(gx+200,gy),(50,50,50),-1)
                for i in range(1, len(carefulness_history)):
                    pt1 = (gx+(i-1)*4, gy - carefulness_history[i-1])
                    pt2 = (gx+i*4,     gy - carefulness_history[i])
                    cv2.line(frame, pt1, pt2, (0,255,0), 2)
                cv2.putText(frame,"Alertness",(gx,gy-105),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

                # Update shared state
                with state_lock:
                    state.update({
                        "status":      status,
                        "level":       level,
                        "score":       max(0, current_score),
                        "ear":         round(avg_ear, 3),
                        "pitch":       round(pitch, 3),
                        "yaw":         round(nose_ratio, 2),
                        "lip":         round(lip_dist, 4),
                        "eye_warns":   eye_warning_count,
                        "yawn_warns":  yawn_warning_count,
                        "head_warns":  head_warning_count,
                        "yawn_msg":    yawn_msg,
                        "tilt_msg":    tilt_msg,
                        "fps":         fps,
                        "session_secs": int(time.time() - session_start),
                    })
        else:
            cv2.putText(frame, "No face detected", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"FPS: {fps}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            output_frame = buf.tobytes()

    cap.release()

# ── FLASK ROUTES ──────────────────────────────────────────────────────────────
def generate():
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame = output_frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with state_lock:
        return jsonify(dict(state))

@app.route('/')
@app.route('/')
def index():
    html_path = os.path.join(os.path.dirname(__file__), 'dashboard (1).html')
    if not os.path.exists(html_path):
        # fallback: search for any html file
        for f in os.listdir(os.path.dirname(__file__)):
            if f.endswith('.html'):
                html_path = os.path.join(os.path.dirname(__file__), f)
                break
    return render_template_string(open(html_path).read())

if __name__ == '__main__':
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("\n  DriveSafe AI is running!")
    print("  Open your browser at:  http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
