# identify_missing_person.py

from src.detection.face_detector import detect_faces
from src.recognition.recognizer_utils import extract_embeddings, compare_faces_cosine
from src.recognition.db_manager import get_all_embeddings, store_embedding

def identify_missing_person(image_np):
    """
    Identifies a missing person from an uploaded image by comparing face embeddings 
    with those stored in the 'missing_persons' collection.

    Args:
        image_np (numpy.ndarray): The uploaded image as a NumPy array.

    Returns:
        dict: A dictionary containing the result of the identification process.
              - If a match is found: {"message": "Match found: [name]", "image_base64": [image_base64]}
              - If no match is found: {"message": "No match found. Added to missing persons: [name]"}
              - If an error occurs: {"error": "Error during identification", "details": [error_details]}
    """
    try:
        image_with_rects, faces = detect_faces(image_np.copy())
        if not faces:
            return {"message": "No faces detected."}

        embeddings = [extract_embeddings(face) for face in faces]
        if not embeddings:
            return {"message": "No valid face embeddings."}

        missing_persons_embeddings = get_all_embeddings("missing_persons")

        for embedding in embeddings:
            match = compare_faces_cosine(embedding, missing_persons_embeddings)
            if match:
                return {"message": f"Match found: {match['name']}", "image_base64": match['image_base64']}

        # No match found, add to missing_persons
        new_person_name = "Unknown_Missing_Person"  # Or generate a unique name
        store_embedding(new_person_name, embeddings[0], image_np, "missing_persons")
        return {"message": f"No match found. Added to missing persons: {new_person_name}"}

    except Exception as e:
        return {"error": "Error during identification", "details": str(e)}