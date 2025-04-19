from src.detection.face_detector import detect_faces
from src.recognition.recognizer_utils import extract_embeddings, cosine_similarity
from src.recognition.db_manager import get_all_embeddings, store_embedding
import numpy as np
from typing import Optional, Dict, Any

def identify_missing_person(
    image_np: np.ndarray,
    name: str = "Unknown",
    age: Optional[int] = None,
    lost_location: Optional[str] = None,
    description: Optional[str] = None,
    contact: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identifies a missing person from an uploaded image by comparing face embeddings
    with those stored in the 'missing_persons' collection. If no match is found,
    it adds the person with provided details.

    Returns:
        dict: Result of the identification process.
    """
    try:
        image_with_rects, faces = detect_faces(image_np.copy())
        print(f"✅ Detected {len(faces)} face(s) in the image.")

        if not faces:
            return {"message": "No faces detected."}

        embeddings = [extract_embeddings(face) for face in faces]
        embeddings = [emb for emb in embeddings if emb is not None]

        if not embeddings:
            return {"message": "No valid face embeddings extracted."}

        print("✅ Extracted embeddings. Fetching missing person records...")
        missing_persons_data = get_all_embeddings("missing_persons")

        best_match = None
        best_score = -1.0

        for person in missing_persons_data:
            stored_embedding = person.get("embedding")
            if stored_embedding:
                score = cosine_similarity(stored_embedding, embeddings[0])
                print(f"🔍 Compared with {person['name']} | Similarity Score: {score:.2f}")
                if score > best_score:
                    best_score = score
                    best_match = person

        if best_score >= 0.80 and best_match:
            print(f"✅ Best match found: {best_match['name']} with score {best_score:.2f}")
            return {
                "message": f"Match found: {best_match['name']}",
                "additional_info": {
                    "name": best_match.get("name"),
                    "age": best_match.get("age"),
                    "lost_location": best_match.get("lost_location"),
                    "description": best_match.get("description"),
                    "contact": best_match.get("contact"),
                }
            }

        # If no match found, register the person
        print(f"❌ No match found. Registering {name} to missing_persons...")
        store_embedding(
            name=name,
            embedding=embeddings[0],
            image_np=image_np,
            collection_name="missing_persons",
            age=age,
            lost_location=lost_location,
            description=description,
            contact=contact
        )
        return {"message": f"No match found. Added to missing persons: {name}"}

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return {"error": "Error during identification", "details": str(e)}
