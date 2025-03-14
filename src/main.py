import cv2
import os
from detection.face_detector import detect_faces 
from recognition.face_recognizer import extract_embeddings, compare_embeddings
from recognition.db_manager import store_embedding
from recognition.add_missing_person import add_missing_person
from recognition.identify_missing_person import identify_missing_person

def store_faces(image_path):
    """Detects faces and stores embeddings in the database."""
    image_with_faces, faces = detect_faces(image_path)
    
    if faces:
        embeddings = extract_embeddings(faces)
        if embeddings:
            print(f"✅ Successfully extracted embeddings for {len(embeddings)} faces.")
            for i, (face, embedding) in enumerate(zip(faces, embeddings)):
                face_id = f"registered_face_{i+1}"
                store_embedding(face_id, "Unknown", embedding, image_path, collection_name="registered_faces")
                cv2.imshow(f"Face {i+1}", face)
            cv2.imshow("Detected Faces", image_with_faces)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("⚠️ No embeddings extracted.")
    else:
        print("⚠️ No faces detected.")

def main():
    while True:
        print("\n🛠 Select an Option:")
        print("1️⃣ Register Known Faces")
        print("2️⃣ Add Missing Person")
        print("3️⃣ Identify Missing Person")
        print("4️⃣ Exit")
        
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            image_path = input("Enter path of image containing known faces: ").strip()
            store_faces(image_path)

        elif choice == "2":
            image_path = input("Enter missing person's image path: ").strip()
            missing_person_name = input("Enter missing person's name: ").strip()
            add_missing_person(image_path, missing_person_name)

        elif choice == "3":
            image_path = input("Enter the image path to identify missing person: ").strip()
            identify_missing_person(image_path)

        elif choice == "4":
            print("👋 Exiting the program.")
            break

        else:
            print("❌ Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
