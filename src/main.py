import cv2
import os
from detection.face_detector import detect_faces 
from recognition.face_recognizer import extract_embeddings

# Define image path
image_path = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\data\test_multiface.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print(f"❌ Error: Image not found at {image_path}")
else:
    print(f"✅ Using image: {image_path}")

    # Detect faces 
    faces = detect_faces(image_path)
    
    # extract embeddings
    embeddings = extract_embeddings(faces)

    if embeddings:
        print(f"✅ Successfully extracted embeddings for {len(embeddings)} faces.")
    else:
        print("⚠️ No embeddings extracted.")
