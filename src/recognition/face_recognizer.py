import cv2
import numpy as np
from deepface import DeepFace

def extract_embeddings(faces):
    """
    Extracts embeddings for each detected face.

    Args:
    - faces (list of numpy arrays): Cropped face images.

    Returns:
    - embeddings_list (list): List of face embeddings.
    """
    embeddings_list = []

    for i, face in enumerate(faces):
        try:
            # Extract embedding
            embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embeddings_list.append(embedding)
        except Exception as e:
            print(f"⚠️ Error extracting embedding for face {i+1}: {e}")

    return embeddings_list
