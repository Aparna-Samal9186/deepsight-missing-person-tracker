from pymongo import MongoClient
import datetime

def setup_database():
    """Initializes the MongoDB database and collections if they don't exist."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["deepsight_db"]
    
    # Check and create collections if they don't exist
    collections = db.list_collection_names()
    if "registered_faces" not in collections:
        db.create_collection("registered_faces")
        print("✅ Created 'registered_faces' collection.")
    if "missing_persons" not in collections:
        db.create_collection("missing_persons")
        print("✅ Created 'missing_persons' collection.")
    
    # Optional: Insert sample face to test setup
    sample_face = {
        "_id": "sample_face_1",
        "name": "Test User",
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # Example embedding
        "image_path": "path/to/sample.jpg",
        "timestamp": datetime.datetime.utcnow()
    }
    
    if db.registered_faces.count_documents({"_id": "sample_face_1"}) == 0:
        db.registered_faces.insert_one(sample_face)
        print("✅ Inserted a sample face into 'registered_faces'.")
    else:
        print("⚠️ Sample face already exists in 'registered_faces'.")
    
    print("🎉 Database setup complete!")

if __name__ == "__main__":
    setup_database()
