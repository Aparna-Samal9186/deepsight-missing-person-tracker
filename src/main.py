import cv2
import os
from detection.face_detector import detect_faces 
from recognition.face_recognizer import extract_embeddings

# Define image path
image_path = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\data\test_multiface.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print(f"❌ Error: Image not found at {image_path}")
else:
    print(f"✅ Using image: {image_path}")

    # Detect faces and get image with bounding boxes
    image_with_faces, faces = detect_faces(image_path)
    
    if faces:
        # Extract embeddings
        embeddings = extract_embeddings(faces)

        if embeddings:
            print(f"✅ Successfully extracted embeddings for {len(embeddings)} faces.")

            # Show detected faces separately and print embeddings
            for i, face in enumerate(faces):
                print(f"🟣 Face {i+1} Embedding: {embeddings[i][:5]}... (truncated for display)")
                cv2.imshow(f"Face {i+1}", face)

            # Show image with bounding boxes
            cv2.imshow("Detected Faces", image_with_faces)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("⚠️ No embeddings extracted.")
    else:
        print("⚠️ No faces detected.")
