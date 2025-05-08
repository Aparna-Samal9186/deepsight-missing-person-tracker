from pymongo import MongoClient
import datetime
import cv2
import base64
import numpy as np
import bcrypt
from bson import ObjectId
from typing import Optional, Union, Dict, List
from sklearn.metrics.pairwise import cosine_similarity

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["deepsight_db"]

# Collections
registered_faces = db["registered_faces"]
missing_persons = db["missing_persons"]
users = db["users"]

# ------------------- Helper Functions -------------------

def convert_image_to_base64(image_np: np.ndarray) -> str:
    """Convert a NumPy image array to a base64-encoded JPEG string."""
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")

def get_next_face_id(collection_name: str = "registered_faces") -> str:
    """Generate the next sequential face ID based on existing documents."""
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    prefix = "registered_face_" if collection_name == "registered_faces" else "missing_face_"

    max_id = 0
    for doc in collection.find({}, {"_id": 1}):
        _id = doc.get("_id", "")
        if _id.startswith(prefix):
            try:
                num = int(_id.split("_")[-1])
                max_id = max(max_id, num)
            except ValueError:
                continue

    return f"{prefix}{max_id + 1}"


# ------------------- Embedding Storage & Retrieval -------------------

def store_embedding(
    name: str,
    embedding: Union[np.ndarray, list],
    image_np: np.ndarray,
    collection_name: str,
    age: Optional[int] = None,
    lost_location: Optional[str] = None,
    description: Optional[str] = None,
    contact: Optional[str] = None
) -> Optional[str]:
    """Store face embedding with additional metadata and default values."""
    if embedding is None or len(embedding) == 0:
        print(f"⚠️ No valid embeddings found for {name}. Skipping storage in {collection_name}.")
        return None

    # Select collection
    collection = missing_persons if collection_name == "missing_persons" else registered_faces
    face_id = get_next_face_id(collection_name)
    image_base64 = convert_image_to_base64(image_np)

    document = {
        "_id": face_id,
        "name": name or "Unknown",
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        "image_base64": image_base64,
        "timestamp": datetime.datetime.utcnow(),
    }
    # Only include non-None optional fields for missing persons
    if collection_name == "missing_persons":
        if age is not None:
            document["age"] = age
        if lost_location is not None:
            document["lost_location"] = lost_location
        if description is not None:
            document["description"] = description
        if contact is not None:
            document["contact"] = contact

    try:
        result = collection.insert_one(document)
        if result.inserted_id:
            print(f"✅ Stored embedding for {name} in {collection_name} with ID: {face_id}")
            return face_id
    except Exception as e:
        print(f"❌ Error storing embedding for {name} in {collection_name}: {e}")

    return None

def get_all_embeddings(collection_name: str = "registered_faces") -> List[Dict]:
    """Retrieve all embeddings with metadata from the specified collection."""
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    documents = list(collection.find({}, {
        "_id": 1,
        "name": 1,
        "embedding": 1,
        "image_base64": 1,
        "age": 1,
        "lost_location": 1,
        "description": 1,
        "contact": 1
    }))

    if not documents:
        print(f"⚠️ No embeddings found in {collection_name}.")
    return documents

# Serialize MongoDB documents for JSON response
def serialize_person(person):
    person["_id"] = str(person["_id"])
    if "timestamp" in person and person["timestamp"]:
        person["timestamp"] = person["timestamp"].isoformat()  # convert datetime to string
    return person

def get_all_missing_persons():
    persons = list(missing_persons.find())
    return [serialize_person(p) for p in persons]

def get_dashboard_stats():
    total_identifications = missing_persons.count_documents({})  # Total entries in missing_persons
    found_count = missing_persons.count_documents({"status": "found"})
    active_cases = missing_persons.count_documents({"status": {"$ne": "found"}})  # Cases not marked as found

    return {
        "totalIdentifications": total_identifications,
        "foundCount": found_count,
        "activeCases": active_cases,
    }

# ------------------- Matching Logic -------------------

def euclidean_distance(e1: list, e2: list) -> float:
    return np.linalg.norm(np.array(e1) - np.array(e2))

def cosine_similarity_score(e1: list, e2: list) -> float:
    """Returns cosine similarity between two vectors (range: -1 to 1)."""
    return cosine_similarity([e1], [e2])[0][0]

def combined_similarity(embedding1: list, embedding2: list, cosine_weight=0.7, euclidean_weight=0.3) -> float:
    """
    Calculate a combined similarity score using both cosine similarity and Euclidean distance.

    Args:
        embedding1 (list): The first embedding vector.
        embedding2 (list): The second embedding vector.
        cosine_weight (float): Weight for cosine similarity (0 to 1).
        euclidean_weight (float): Weight for the normalized inverse of Euclidean distance (0 to 1).

    Returns:
        float: The combined similarity score.
    """
    # Cosine Similarity
    cosine_sim = cosine_similarity_score(embedding1, embedding2)

    # Euclidean Distance (convert to similarity)
    euclidean_dist = euclidean_distance(embedding1, embedding2)
    # Normalize Euclidean distance to a similarity score (higher is better, between 0 and 1)
    max_dist = 10  #  adjust based on expected max distance in your embedding space.
    euclidean_sim = 1 - (euclidean_dist / max_dist)
    euclidean_sim = max(0, euclidean_sim)  # Ensure non-negative

    # Combine the two measures
    combined_score = (cosine_weight * cosine_sim) + (euclidean_weight * euclidean_sim)
    return combined_score
    
def find_closest_match(embedding: list, collection_name: str = "registered_faces") -> Optional[Dict]:
    """Find the most similar face using combined similarity from the specified collection."""
    all_entries = get_all_embeddings(collection_name)
    best_match = None
    best_similarity = -1.0

    for entry in all_entries:
        stored_embedding = entry.get("embedding")
        if stored_embedding:
            similarity = combined_similarity(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

    THRESHOLD = 0.75  # Combined similarity threshold
    return best_match if best_similarity >= THRESHOLD else None

# ------------------- User Management -------------------

def create_user(username: str, password: str) -> Dict[str, str]:
    """Register a new user with hashed password."""
    if users.find_one({"username": username}):
        return {"error": "User already exists."}

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users.insert_one({"username": username, "password": hashed_pw})
    return {"message": "User registered successfully!"}

def verify_user(username: str, password: str) -> Dict[str, str]:
    """Verify user login credentials."""
    user = users.find_one({"username": username})
    if not user:
        return {"error": "User not found."}

    if bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return {"message": "Login successful!"}
    else:
        return {"error": "Incorrect password."}