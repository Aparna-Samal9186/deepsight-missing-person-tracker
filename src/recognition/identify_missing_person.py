from src.detection.face_detector import detect_faces
from src.recognition.recognizer_utils import extract_embeddings
from src.recognition.db_manager import get_all_embeddings, store_embedding, cosine_similarity_score
import numpy as np
from typing import Optional, Dict, Any
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

# Re-establish MongoDB client (if needed in this context)
client = MongoClient("mongodb://localhost:27017/")
db = client["deepsight_db"]
missing_persons_collection = db["missing_persons"]
missing_persons_collection1 = db["registered_faces"]

def identify_missing_person(
    image_np: np.ndarray,
    name: str = "Unknown",
    age: Optional[int] = None,
    lost_location: Optional[str] = None,
    description: Optional[str] = None,
    contact: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identifies a missing person from an uploaded image. If no match is found
    in the existing database based on the similarity threshold, the uploaded
    image and details are stored as a new missing person.

    Returns:
        dict: Result of the identification process.
    """
    try:
        # 1. Detect Faces
        image_with_rects, faces = detect_faces(image_np.copy())
        print(f"✅ Detected {len(faces)} face(s) in the image.")

        if not faces:
            return {"message": "No faces detected in the uploaded image."}

        # 2. Extract Embeddings
        embeddings = [extract_embeddings(face) for face in faces]
        embeddings = [emb for emb in embeddings if emb is not None]

        if not embeddings:
            return {"message": "No valid face embeddings extracted from the uploaded image."}

        input_embedding = embeddings[0]
        print("✅ Extracted embedding from the uploaded image. Fetching existing missing person records...")

        # 3. Fetch Existing Embeddings from Missing Persons Collection
        missing_persons_data = get_all_embeddings("missing_persons")

        best_match = None
        best_similarity = -1.0  # Cosine similarity range: -1 to 1

        # 4. Find Closest Match (if there are existing missing persons)
        if missing_persons_data:
            for person in missing_persons_data:
                stored_embedding = person.get("embedding")
                if stored_embedding:
                    similarity = cosine_similarity_score(input_embedding, stored_embedding)
                    print(f"🔍 Compared with {person['name']} | Similarity: {similarity:.4f}")

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person

            MATCH_THRESHOLD = 0.852  # <--- ADJUST THIS VALUE BASED ON YOUR TESTING

            if best_similarity >= MATCH_THRESHOLD and best_match:
                print(f"✅ Match found: {best_match['name']} | Similarity: {best_similarity:.4f}")
                try:
                    update_result = missing_persons_collection.update_one(
                        {"_id": best_match["_id"]},
                        {"$set": {"status": "found"}}
                    )
                    if update_result.modified_count > 0:
                        print(f"📝 Updated status to 'found' for ID: {best_match['_id']}")
                    else:
                        print(f"⚠️ Failed to update status for ID: {best_match['_id']}. No documents matched the query.")
                except Exception as update_error:
                    logger.error(f"⚠️ Failed to update status: {update_error}")

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
            else:
                # 5. Register as Missing if no match found above threshold
                print(f"❌ No match found (similarity: {best_similarity:.4f} < {MATCH_THRESHOLD}). Registering {name} to missing_persons...")
                store_embedding(
                    name=name,
                    embedding=input_embedding,
                    image_np=image_np,
                    collection_name="missing_persons",
                    age=age,
                    lost_location=lost_location,
                    description=description,
                    contact=contact
                )
                return {"message": f"No match found. Added to missing persons: {name}"}

        else:
            # 6. If no existing missing persons in the database, directly store the uploaded person
            print("⚠️ No existing missing persons in the database. Adding the uploaded person.")
            store_embedding(
                name=name,
                embedding=input_embedding,
                image_np=image_np,
                collection_name="missing_persons",
                age=age,
                lost_location=lost_location,
                description=description,
                contact=contact
            )
            return {"message": f"No existing missing persons found. Added {name} to missing persons."}

    except Exception as e:
        logger.error(f"❌ Exception occurred during identification: {e}")
        return {"error": "Error during identification", "details": str(e)}

# For Webcam  
def identify_missingPerson(
    image_np: np.ndarray,
    name: str = "Unknown",
    age: Optional[int] = None,
    lost_location: Optional[str] = None,
    description: Optional[str] = None,
    contact: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identifies a missing person from an uploaded image. If no match is found
    in the existing database based on the similarity threshold, the uploaded
    image and details are stored as a new missing person.

    Returns:
        dict: Result of the identification process.
    """
    try:
        # 1. Detect Faces
        image_with_rects, faces = detect_faces(image_np.copy())
        print(f"✅ Detected {len(faces)} face(s) in the image.")

        if not faces:
            return {"message": "No faces detected in the uploaded image."}

        # 2. Extract Embeddings
        embeddings = [extract_embeddings(face) for face in faces]
        embeddings = [emb for emb in embeddings if emb is not None]

        if not embeddings:
            return {"message": "No valid face embeddings extracted from the uploaded image."}

        input_embedding = embeddings[0]
        print("✅ Extracted embedding from the uploaded image. Fetching existing missing person records...")

        # 3. Fetch Existing Embeddings from Missing Persons Collection
        missing_persons_data = get_all_embeddings("registered_faces")

        best_match = None
        best_similarity = -1.0  # Cosine similarity range: -1 to 1

        # 4. Find Closest Match (if there are existing missing persons)
        if missing_persons_data:
            for person in missing_persons_data:
                stored_embedding = person.get("embedding")
                if stored_embedding:
                    similarity = cosine_similarity_score(input_embedding, stored_embedding)
                    print(f"🔍 Compared with {person['name']} | Similarity: {similarity:.4f}")

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person

            MATCH_THRESHOLD = 0.6  # <--- ADJUST THIS VALUE BASED ON YOUR TESTING

            if best_similarity >= MATCH_THRESHOLD and best_match:
                print(f"✅ Match found: {best_match['name']} | Similarity: {best_similarity:.4f}")
                try:
                    update_result = missing_persons_collection1.update_one(
                        {"_id": best_match["_id"]},
                        {"$set": {"status": "found"}}
                    )
                    if update_result.modified_count > 0:
                        print(f"📝 Updated status to 'found' for ID: {best_match['_id']}")
                    else:
                        print(f"⚠️ Failed to update status for ID: {best_match['_id']}. No documents matched the query.")
                except Exception as update_error:
                    logger.error(f"⚠️ Failed to update status: {update_error}")

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
            else:
                # 5. Register as Missing if no match found above threshold
                print(f"❌ No match found (similarity: {best_similarity:.4f} < {MATCH_THRESHOLD}). Registering {name} to missing_persons...")
                store_embedding(
                    name=name,
                    embedding=input_embedding,
                    image_np=image_np,
                    collection_name="registered_faces",
                    age=age,
                    lost_location=lost_location,
                    description=description,
                    contact=contact
                )
                return {"message": f"No match found. Added to missing persons: {name}"}

        else:
            # 6. If no existing missing persons in the database, directly store the uploaded person
            print("⚠️ No existing missing persons in the database. Adding the uploaded person.")
            store_embedding(
                name=name,
                embedding=input_embedding,
                image_np=image_np,
                collection_name="registered_faces",
                age=age,
                lost_location=lost_location,
                description=description,
                contact=contact
            )
            return {"message": f"No existing missing persons found. Added {name} to missing persons."}

    except Exception as e:
        logger.error(f"❌ Exception occurred during identification: {e}")
        return {"error": "Error during identification", "details": str(e)}