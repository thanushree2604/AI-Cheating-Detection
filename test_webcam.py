from ultralytics import YOLO
import cv2

# Load YOLOv8 model (downloads yolov8n.pt automatically first time)
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model.predict(frame, imgsz=640, conf=0.5)
    annotated_frame = results[0].plot()

    # Show the webcam feed
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
