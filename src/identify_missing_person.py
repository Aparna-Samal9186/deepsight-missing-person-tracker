import cv2
import os
from detection.face_detector import detect_faces
from recognition.face_recognizer import extract_embeddings, compare_embeddings
from recognition.db_manager import get_all_embeddings

def identify_missing_person(image_path):
    """Identifies missing persons by comparing detected faces with registered faces."""
    print(f"✅ Processing image: {image_path}")
    
    image_with_faces, faces = detect_faces(image_path)
    if not faces:
        print("⚠️ No faces detected.")
        return
    
    embeddings = extract_embeddings(faces)
    if not embeddings:
        print("⚠️ No embeddings extracted.")
        return
    
    registered_faces = get_all_embeddings("registered_faces")
    
    for i, embedding in enumerate(embeddings):
        face_id = f"face_{i+1}"
        print(f"🔍 Comparing {face_id}...")
        
        matches = compare_embeddings(embedding, collection_name="registered_faces")
        
        if matches:
            matched_name, similarity = matches[0]
            print(f"🚨 Match Found! {face_id} matches registered person: {matched_name} (Similarity: {similarity:.2f})")
            
            # Retrieve and show the registered person's image with bounding box
            for entry in registered_faces:
                if entry["name"] == matched_name:
                    registered_image_path = entry["image_path"]
                    face_bbox = entry.get("bbox")  # Get bounding box (x, y, w, h)
                    
                    if os.path.exists(registered_image_path):
                        registered_image = cv2.imread(registered_image_path)
                        
                        if face_bbox:
                            x, y, w, h = face_bbox
                            cv2.rectangle(registered_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        
                        cv2.imshow(f"Matched Registered Person: {matched_name}", registered_image)
                    break
        else:
            print(f"⚠️ {face_id} does not match any registered person.")
    
    cv2.imshow("Detected Faces in Missing Person Image", image_with_faces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the identification
missing_person_image = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\data\test_image.jpg"
identify_missing_person(missing_person_image)
