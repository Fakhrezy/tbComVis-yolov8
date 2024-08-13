from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model_path = "best.pt"
model = YOLO(model_path)

# Path to the video file
video_path = "p.mp4"  # Replace with your video path

# Open video capture
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(frame_rgb)

    # Process results
    for result in results:
        boxes = result.boxes  # Detection boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Box coordinates [x1, y1, x2, y2]
            conf = box.conf[0].cpu().numpy()  # Confidence score
            cls = int(box.cls[0].cpu().numpy())  # Detected class

            # Draw detection boxes and labels
            if cls in range(len(model.names)):  # Check if class index is valid
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    # Display frame with detections
    cv2.imshow("Detected Objects", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()