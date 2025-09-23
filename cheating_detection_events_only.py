import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
from playsound import playsound

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "yolov8n.pt"
LOGS_DIR = "logs"
SNAPSHOTS_DIR = "snapshots"
RECORDS_DIR = "records"

CONF_THRESHOLD = 0.5
FPS = 20.0
EVENT_DURATION = 5  # seconds of recording when object detected
ALERT_SOUNDS = {
    "cell phone": "alert_cell.wav",
    "book": "alert_book.wav",
    "laptop": "alert_laptop.wav"
}

# Create folders if not exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)

# ================================
# FUNCTIONS
# ================================
def initialize_model(model_path):
    return YOLO(model_path)

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("❌ Error: Could not open webcam")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height

def log_detection(detected_objects):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOGS_DIR, "cheating_log.txt")
    with open(log_file, "a") as f:
        f.write(f"{timestamp}: Detected {', '.join(detected_objects)}\n")
    return timestamp

def save_snapshot(frame, timestamp):
    snapshot_path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, frame)

def save_event_video(frames, timestamp, width, height):
    clip_path = os.path.join(RECORDS_DIR, f"event_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(clip_path, fourcc, FPS, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def play_alert_sound(detected_objects):
    first_obj = detected_objects[0]
    if first_obj in ALERT_SOUNDS:
        try:
            playsound(ALERT_SOUNDS[first_obj])
        except:
            print(f"⚠️ Could not play sound for {first_obj}")

def detect_objects(model, frame):
    results = model(frame, stream=True, conf=CONF_THRESHOLD)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if label in ALERT_SOUNDS:
                detected_objects.append(label)
    return frame, detected_objects

# ================================
# MAIN FUNCTION
# ================================
def main():
    print("🚀 Starting Cheating Detection... Press 'q' to quit.")

    model = initialize_model(MODEL_PATH)
    cap, width, height = initialize_camera()

    recording = False
    event_frames = []
    event_start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = detect_objects(model, frame)

        # When object detected → start recording event video
        if detected_objects and not recording:
            recording = True
            event_start_time = time.time()
            event_frames = []
            timestamp = log_detection(detected_objects)
            save_snapshot(frame, timestamp)
            play_alert_sound(detected_objects)
            print(f"🎥 Recording started for event at {timestamp}...")

        # If currently recording, save frames until EVENT_DURATION
        if recording:
            event_frames.append(frame.copy())
            if time.time() - event_start_time >= EVENT_DURATION:
                save_event_video(event_frames, timestamp, width, height)
                print(f"💾 Event video saved at {timestamp}")
                recording = False

        cv2.imshow("Cheating Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detection ended. All event videos, snapshots, and logs saved.")

# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    main()
