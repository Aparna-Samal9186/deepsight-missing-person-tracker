# recognizer_utils.py
import numpy as np
from deepface import DeepFace  # Import DeepFace for face recognition
from src.recognition.db_manager import euclidean_distance, cosine_similarity_score  # Import both similarity metrics

def extract_embeddings(face_image):
    """Extract face embeddings using DeepFace."""
    try:
        embedding_obj = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=False) #enforce_detection=False, because face is already detected.
        if embedding_obj and len(embedding_obj) > 0:
            embedding = embedding_obj[0]["embedding"]
            return embedding
        else:
            print("⚠️ DeepFace representation returned an empty list or None.")
            return None
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")
        return None  # Handle errors gracefully


def compare_faces_cosine(embedding, stored_embeddings):
    """Compare a detected face's embedding to stored embeddings using cosine similarity."""
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1
    match = None

    for stored_embedding_data in stored_embeddings:
        stored_embedding = stored_embedding_data.get("embedding")
        if stored_embedding:  # Ensure embedding exists
            similarity = cosine_similarity_score(embedding, stored_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                match = stored_embedding_data

    return match if highest_similarity > 0.6 else None  # Threshold for match