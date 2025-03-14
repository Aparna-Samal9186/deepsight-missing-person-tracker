from pymongo import MongoClient
import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["deepsight_db"]

# Collections
registered_faces = db["registered_faces"]  # Known individuals
missing_persons = db["missing_persons"]  # Missing persons

def get_next_face_id():
    """
    Retrieves the next available face ID for registered faces.
    
    Returns:
    - str: New unique face ID.
    """
    last_face = registered_faces.find_one(
        {}, 
        sort=[("_id", -1)]
    )

    if last_face and last_face["_id"].startswith("registered_face_"):
        last_id = int(last_face["_id"].split("_")[-1])
        new_id = last_id + 1
    else:
        new_id = 1

    return f"registered_face_{new_id}"


def store_embedding(face_id, name, embedding, image_path, collection_name="registered_faces"):
    """
    Stores a face embedding in MongoDB.
    
    Args:
    - face_id (str): Unique identifier for the face.
    - name (str): Name of the person (or "Unknown" for missing persons).
    - embedding (list): Face embedding vector.
    - image_path (str): Path to the image.
    - collection_name (str): Collection to store data in ('registered_faces' or 'missing_persons').

    Returns:
    - str: Inserted document ID.
    """
    collection = registered_faces if collection_name == "registered_faces" else missing_persons

    document = {
        "_id": face_id,
        "name": name,
        "embedding": embedding,
        "image_path": image_path,
        "timestamp": datetime.datetime.utcnow()
    }

    try:
        collection.insert_one(document)
        print(f"✅ Stored embedding for {name} in {collection_name}.")
        return str(document["_id"])
    except Exception as e:
        print(f"⚠️ Error storing embedding: {e}")
        return None

def get_all_embeddings(collection_name="registered_faces"):
    """
    Retrieves all face embeddings from the specified collection.
    
    Args:
    - collection_name (str): 'registered_faces' or 'missing_persons'

    Returns:
    - list: Retrieved embeddings.
    """
    collection = registered_faces if collection_name == "registered_faces" else missing_persons
    return list(collection.find({}, {"_id": 1, "name": 1, "embedding": 1, "image_path": 1}))
