import cv2
import numpy as np
from deepface import DeepFace

from scipy.spatial.distance import cosine
from recognition.db_manager import get_all_embeddings # Load stored embeddings

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


def compare_embeddings(new_embedding, threshold=0.7, collection_name="registered_faces"):
    """
    Compares a new embedding against stored embeddings in MongoDB.
    """
    stored_faces = get_all_embeddings(collection_name)
    matches = []
    
    for entry in stored_faces:
        name, stored_embedding = entry["name"], entry["embedding"]
        similarity = 1 - cosine(new_embedding, stored_embedding)
        
        if similarity >= threshold:
            matches.append((name, similarity))
    
    matches.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
    return matches