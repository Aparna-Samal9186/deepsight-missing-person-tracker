import cv2
import numpy as np
import os
from src.detection.face_detector import detect_faces
from .face_recognizer import extract_embeddings
from .db_manager import store_embedding

def add_missing_person(image, missing_person_name):
    """
    Detects a missing person from an image, extracts embeddings, and stores them in MongoDB.
    
    - image (numpy.ndarray): Image of the missing person.
    - missing_person_name (str): Name of the missing person.
    """
    print(f"✅ Adding missing person: {missing_person_name}")

    # Detect faces
    image_with_faces, faces = detect_faces(image)

    if faces:
        print(f"✅ Detected {len(faces)} face(s), extracting embeddings...")

        embeddings = extract_embeddings(faces)
        print(f"✅ Extracted {len(embeddings)} embedding(s).")

        if embeddings:
            print(f"✅ Extracted embeddings for {len(embeddings)} face(s). Storing in DB...")
            for i, embedding in enumerate(embeddings):
                store_embedding(missing_person_name, embedding, image, collection_name="missing_persons")
                print(f"✅ Storing embedding for {name}: {embedding}")
        else:
            print("⚠️ No embeddings extracted. Data will NOT be stored.")

