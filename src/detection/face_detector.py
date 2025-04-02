# face_detector.py

import cv2
import dlib
import numpy as np
import mediapipe as mp
from src.recognition.recognizer_utils import extract_embeddings, compare_faces
from src.recognition.db_manager import get_all_embeddings

# Load face detector
# detector = dlib.get_frontal_face_detector()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# def process_frame(image):
#     """
#     Detect faces and recognize them in a given image.

#     Args:
#     - image (numpy.ndarray): Input image.

#     Returns:
#     - image (numpy.ndarray): Image with bounding boxes and labels.
#     - results (list): List of detected face information.
#     """
#     if image is None or image.size == 0:
#         return None, {"error": "Invalid image provided."}

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     stored_embeddings = get_all_embeddings()
#     results = []
    
#     for face in faces:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()

#         # Ensure valid face cropping
#         if y < 0 or x < 0 or y+h > image.shape[0] or x+w > image.shape[1]:
#             continue

#         cropped_face = image[y:y+h, x:x+w]
#         embedding = extract_embeddings(cropped_face)

#         match = compare_faces(embedding, stored_embeddings)
#         label = "Unknown" if match is None else match
#         results.append({"label": label, "coordinates": [x, y, w, h]})

#         cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     return image, results

def detect_faces(image):
    if image is None or image.size == 0:
        print("❌ Error: Empty image received in detect_faces.")
        return None, []

    print(f"✅ detect_faces received image with shape: {image.shape}, dtype: {image.dtype}")

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox

                if y < 0 or x < 0 or y + h > image.shape[0] or x + w > image.shape[1]:
                    continue

                cropped_face = image[y:y+h, x:x+w]
                faces.append(cropped_face)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection) # add landmarks
        return image, faces

    except Exception as e:
        print(f"❌ MediaPipe Error: {str(e)}")
        return None, []