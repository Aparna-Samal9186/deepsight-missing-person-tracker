# recognizer_utils.py
import numpy as np
from deepface import DeepFace  # Import DeepFace for face recognition
from src.recognition.db_manager import cosine_similarity  # Import cosine_similarity from db_manager

def extract_embeddings(face_image):
    """Extract face embeddings using DeepFace."""
    try:
        embedding_obj = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=False) #enforce_detection=False, because face is already detected.
        embedding = embedding_obj[0]["embedding"]
        return embedding
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")
        return None  # Handle errors gracefully


def compare_faces_cosine(embedding, stored_embeddings):
    """Compare a detected face's embedding to stored embeddings using cosine similarity."""
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1
    match = None

    for stored_embedding in stored_embeddings:
        if stored_embedding.get("embedding"):  # Ensure embedding exists
            similarity = cosine_similarity(embedding, stored_embedding["embedding"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                match = stored_embedding

    return match if highest_similarity > 0.6 else None  # Threshold for match