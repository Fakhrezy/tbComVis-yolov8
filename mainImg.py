from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
            conf_percent = conf * 100  # Convert to percentage
            label = f"{model.names[cls]} {conf_percent:.0f}%"  # Format as percentage without decimal

            # Set color to bright green (similar to the uploaded image)
            color = (0, 255, 0)
            thickness = 2  # Moderate thickness

            # Draw rectangle with desired thickness
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)

            # Calculate label size for better placement
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw filled rectangle for label background
            cv2.rectangle(
                image_rgb,
                (x1, y1),
                (x1 + label_width, y1 - label_height - baseline),
                color,
                cv2.FILLED,
            )

            # Draw label text inside the box
            cv2.putText(
                image_rgb,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black color for text
                2,
            )

# Display image with detections using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()

# Save image with detections
output_path = "output_image.jpg"  # Replace with your desired output path
cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
