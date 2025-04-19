from pymongo import MongoClient
import datetime
import cv2
import base64
import numpy as np
import bcrypt
from typing import Optional, Union, Dict, List

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
    """Generate the next sequential face ID based on the collection."""
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    last_face = collection.find_one(sort=[("_id", -1)])

    prefix = "registered_face_" if collection_name == "registered_faces" else "missing_face_"
    
    if last_face and last_face.get("_id", "").startswith(prefix):
        try:
            last_id = int(last_face["_id"].split("_")[-1])
            return f"{prefix}{last_id + 1}"
        except (ValueError, IndexError):
            pass
    
    return f"{prefix}1"

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
    """Store face embedding with additional metadata."""
    if embedding is None or len(embedding) == 0:
        print(f"⚠️ No valid embeddings found for {name}. Skipping storage.")
        return None

    collection = missing_persons if collection_name == "missing_persons" else registered_faces
    face_id = get_next_face_id(collection_name)
    image_base64 = convert_image_to_base64(image_np)

    document = {
        "_id": face_id,
        "name": name,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        "image_base64": image_base64,
        "timestamp": datetime.datetime.utcnow(),
        "age": age,
        "lost_location": lost_location,
        "description": description,
        "contact": contact
    }

    try:
        result = collection.insert_one(document)
        if result.inserted_id:
            print(f"✅ Stored embedding for {name} in {collection_name} with ID: {face_id}")
            return face_id
    except Exception as e:
        print(f"❌ Error storing embedding for {name}: {e}")
    
    return None

def get_all_embeddings(collection_name: str = "registered_faces") -> List[Dict]:
    """Retrieve all embeddings with metadata."""
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

# ------------------- Matching Logic -------------------

def cosine_similarity(embedding1: list, embedding2: list) -> float:
    """Compute cosine similarity between two embeddings."""
    e1, e2 = np.array(embedding1), np.array(embedding2)
    norm_e1, norm_e2 = np.linalg.norm(e1), np.linalg.norm(e2)

    if norm_e1 == 0 or norm_e2 == 0:
        return 0.0
    return np.dot(e1, e2) / (norm_e1 * norm_e2)

def find_closest_match(embedding: list, collection_name: str = "registered_faces") -> Optional[Dict]:
    """Find the most similar face using cosine similarity."""
    all_entries = get_all_embeddings(collection_name)
    best_match = None
    best_score = -1

    for entry in all_entries:
        stored_embedding = entry.get("embedding")
        if stored_embedding:
            score = cosine_similarity(embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_match = entry

    return best_match if best_score > 0.8 else None

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
    return {"error": "Invalid credentials."}
