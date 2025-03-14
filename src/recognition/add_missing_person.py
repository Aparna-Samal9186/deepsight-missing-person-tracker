import cv2
import os
from detection.face_detector import detect_faces
from recognition.face_recognizer import extract_embeddings
from recognition.db_manager import store_embedding

def add_missing_person(image_path, missing_person_name):
    """
    Detects a missing person, extracts embeddings, and stores them in the database.
    
    Args:
    - image_path (str): Path to the missing person's image.
    - missing_person_name (str): Name of the missing person.

    Returns:
    - None
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    print(f"✅ Adding missing person: {missing_person_name}")

    # Detect faces
    image_with_faces, faces = detect_faces(image_path)

    if faces:
        # Extract embeddings
        embeddings = extract_embeddings(faces)

        if embeddings:
            print(f"✅ Successfully extracted embeddings for {len(embeddings)} missing persons.")

            # Store each detected face in the "missing_persons" collection
            for i, embedding in enumerate(embeddings):
                face_id = f"{missing_person_name}_face_{i+1}"
                store_embedding(face_id, missing_person_name, embedding, image_path, collection_name="missing_persons")

            print(f"✅ {missing_person_name} added to missing persons database!")
        else:
            print("⚠️ No embeddings extracted.")
    else:
        print("⚠️ No faces detected in the image.")

# Uncomment the below lines to run this file independently
# if __name__ == "__main__":
#     test_image_path = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\data\test_image.jpg"
#     add_missing_person(test_image_path, "Elon Musk")
