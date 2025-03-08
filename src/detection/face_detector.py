import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace

# Initialize MTCNN detector
detector = MTCNN()

def detect_faces(image_path):
    """
    Detects multiple faces in an image and returns cropped face regions.

    Args:
    - image_path (str): Path to the input image.

    Returns:
    - faces (list of numpy arrays): List of cropped face images.
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Unable to load image!")
        return []

    # Convert image to RGB (MTCNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detector.detect_faces(image_rgb)
    
    faces = []
    
    for i, detection in enumerate(detections):
        x, y, width, height = detection['box']

        # Ensure valid bounding box dimensions
        x, y = max(0, x), max(0, y)
        width, height = max(1, width), max(1, height)

        # Extract face region
        face_rgb = image_rgb[y:y + height, x:x + width]

        # Convert face back to BGR (DeepFace expects BGR)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        # Resize face to match model input size
        face_bgr = cv2.resize(face_bgr, (160, 160))

        faces.append(face_bgr)

    print(f"✅ Detected {len(faces)} faces.")
    return faces


