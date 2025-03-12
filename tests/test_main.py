# import dlib

# predictor_path = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\models\shape_predictor_68_face_landmarks.dat"

# try:
#     predictor = dlib.shape_predictor(predictor_path)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")

import cv2
from detection.face_detector import detect_faces  # Ensure this function returns cropped faces
from recognition.face_recognizer import extract_embeddings
from db_manager import connect_db  # Ensure this function is implemented

# Path to test image (update with actual image path)
image_path = "data/test_image.jpg"

# Load image
image = cv2.imread(image_path)

# Detect faces
faces, _ = detect_faces(image)  # Ensure this function returns cropped face images

# Dummy names list (for now, assign "Unknown")
names = ["Unknown"] * len(faces)

if len(faces) > 0:
    # Extract embeddings and store them in MongoDB
    embeddings = extract_embeddings(faces, names, image_path)
    
    if embeddings:
        print("✅ Face embeddings successfully extracted and stored in MongoDB!")
    else:
        print("❌ No embeddings extracted.")
else:
    print("⚠️ No faces detected in the image.")
