# db_manager.py
from pymongo import MongoClient
import datetime
import cv2
import base64
import numpy as np

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["deepsight_db"]

# Collections
registered_faces = db["registered_faces"]
missing_persons = db["missing_persons"]

for doc in missing_persons.find():
    print(doc)

def convert_image_to_base64(image_np):
    """Converts a NumPy image array to Base64 string."""
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")

def get_next_face_id(collection_name="registered_faces"):
    """Generates a unique face ID."""
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    last_face = collection.find_one({}, sort=[("_id", -1)])

    prefix = "registered_face_" if collection_name == "registered_faces" else "missing_face_"

    if last_face and last_face.get("_id", "").startswith(prefix):
        try:
            last_id = int(last_face["_id"].split("_")[-1])
            new_id = last_id + 1
        except ValueError:
            new_id = 1
    else:
        new_id = 1

    return f"{prefix}{new_id}"

def store_embedding(name, embedding, image_np, collection_name):
    """Stores a face embedding in MongoDB."""
    collection = missing_persons if collection_name == "missing_persons" else registered_faces

    # Ensure embedding is not None or empty
    if embedding is None or len(embedding) == 0:
        print(f"⚠️ No valid embeddings found for {name}. Skipping storage.")
        return None

    face_id = get_next_face_id(collection_name)
    image_base64 = convert_image_to_base64(image_np)

    document = {
        "_id": face_id,
        "name": name,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,  # Ensure embedding is stored as a list
        "image_base64": image_base64,
        "timestamp": datetime.datetime.utcnow()
    }

    try:
        insert_result = collection.insert_one(document)
        if insert_result.inserted_id:
            print(f"✅ Stored embedding for {name} in {collection_name} with ID: {insert_result.inserted_id}")
            return face_id
        else:
            print(f"⚠️ MongoDB did not return an inserted ID for {name}.")
    except Exception as e:
        print(f"❌ Error storing embedding for {name}: {e}")
    
    return None

def get_all_embeddings(collection_name="registered_faces"):
    """Retrieves all face embeddings."""
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    embeddings = list(collection.find({}, {"_id": 1, "name": 1, "embedding": 1, "image_base64": 1}))

    if not embeddings:
        print(f"⚠️ No embeddings found in {collection_name}.")
    
    return embeddings

def cosine_similarity(embedding1, embedding2):
    """Computes cosine similarity between two embeddings."""
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)

    # Avoid division by zero
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)

    if norm_e1 == 0 or norm_e2 == 0:
        return 0  # Return 0 if either vector is zero-length

    return np.dot(e1, e2) / (norm_e1 * norm_e2)

def find_closest_match(embedding, collection_name="registered_faces"):
    """Finds the closest face match based on cosine similarity."""
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    all_embeddings = get_all_embeddings(collection_name)

    if not all_embeddings:
        return None

    closest_match = None
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1

    for entry in all_embeddings:
        stored_embedding = entry.get("embedding")
        if stored_embedding:
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_match = entry

    return closest_match if highest_similarity > 0.8 else None  # 0.8 threshold for match
