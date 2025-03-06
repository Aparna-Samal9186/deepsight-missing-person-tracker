import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

def detect_faces(image_path):
    """
    Detects faces in an image using MTCNN.
    Draws bounding boxes and facial landmarks.

    Args:
    - image_path (str): Path to the input image.

    Returns:
    - image (numpy array): Image with detected faces.
    - num_faces (int): Number of faces detected.
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Error: Unable to load image!")
        return None, 0

    # Convert image to RGB (MTCNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detector.detect_faces(image_rgb)
    num_faces = len(detections)

    # Draw bounding boxes and landmarks
    for detection in detections:
        x, y, width, height = detection['box']
        keypoints = detection['keypoints']

        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)

        # Draw key points (eyes, nose, mouth)
        for point in keypoints.values():
            cv2.circle(image, point, 3, (0, 0, 255), -1)

    return image, num_faces
