from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # automatically downloads latest yolov8n.pt if missing
print("YOLOv8 is working!")
