# face_detector.py

import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

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
                mp_drawing.draw_detection(image, detection)  # add landmarks
        return image, faces

    except Exception as e:
        print(f"❌ MediaPipe Error: {str(e)}")
        return None, []