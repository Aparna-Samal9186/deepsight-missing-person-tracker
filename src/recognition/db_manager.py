from pymongo import MongoClient
import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["deepsight_db"]
collection = db["face_embeddings"]

def store_embedding(face_id, name, embedding, image_path):
    """Stores a face embedding in MongoDB."""
    document = {
        "_id": face_id,
        "name": name,
        "embedding": embedding,  # Already a list in main.py
        "image_path": image_path,
        "timestamp": datetime.datetime.utcnow()
    }
    try:
        collection.insert_one(document)
        print(f"✅ Stored embedding for {name} ({face_id})")
    except Exception as e:
        print(f"⚠️ Error storing embedding: {e}")

def get_all_embeddings():
    """Retrieves all face embeddings from MongoDB."""
    return list(collection.find({}, {"_id": 1, "name": 1, "embedding": 1}))
