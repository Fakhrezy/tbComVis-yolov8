from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model_path = "best.pt"
model = YOLO(model_path)

# Path to the image file
image_path = "holeTest.jpg"  # Replace with your image path

# Load image
image = cv2.imread(image_path)

# Convert image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform detection
results = model(image_rgb)

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
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

# Display image with detections
cv2.imshow("Detected Objects", image)

# Save image with detections
output_path = "output_image.jpg"  # Replace with your desired output path
cv2.imwrite(output_path, image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
