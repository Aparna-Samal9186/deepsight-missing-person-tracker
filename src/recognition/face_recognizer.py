# face_recognizer.py
import cv2
import numpy as np
import dlib
from src.recognition.db_manager import get_all_embeddings, find_closest_match
from src.recognition.recognizer_utils import extract_embeddings, compare_faces

# Load face detector
detector = dlib.get_frontal_face_detector()

def recognize_face(input_type="image", image_path=None):
    """
    Recognizes faces from an image or webcam.

    Args:
    - input_type (str): "image" for static images, "webcam" for real-time webcam capture.
    - image_path (str, optional): Path to the image (only used if input_type is "image").
    
    Returns:
    - None
    """
    
    if input_type == "image" and image_path:
        image = cv2.imread(image_path)
        if image is None:
            print("⚠️ Error: Could not load image.")
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Error: Could not open webcam.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Error: Could not read frame.")
                break
            
            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Capture frame
                image = frame.copy()
                break
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Process detected faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print("⚠️ No faces detected.")
        return
    
    stored_embeddings = get_all_embeddings()
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cropped_face = image[y:y+h, x:x+w]
        
        # Extract embedding
        embedding = extract_embeddings(cropped_face)
        
        # Find closest match in the database
        match = find_closest_match(embedding, "registered_faces")
        
        label = "Unknown" if match is None else match["name"]
        
        # Draw label and bounding box
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the results
    cv2.imshow("Recognition Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# recognize_face("image", "path/to/image.jpg")
# recognize_face("webcam")

