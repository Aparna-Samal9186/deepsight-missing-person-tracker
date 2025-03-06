import cv2
import os
from detection.face_detector import detect_faces

# Define image path
image_path = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\data\BillGates.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print(f"❌ Error: Image not found at {image_path}")
else:
    print(f"✅ Using image: {image_path}")

    # Detect faces
    image, num_faces = detect_faces(image_path)

    if image is not None:
        print(f"👤 Number of faces detected: {num_faces}")

        # Show image with detections
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
