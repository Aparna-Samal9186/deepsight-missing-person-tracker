
import cv2
import face_recognition
import numpy as np

def main():
    print("Starting DeepSight: AI-Powered Missing Person Tracker")
    # Load a sample image for demo purposes
    image = cv2.imread('sample_image.jpg')
    if image is None:
        print("Image not found. Please provide a valid image file.")
        return

    # Convert image to RGB (face_recognition uses RGB format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_image)
    print(f"Detected {len(face_locations)} face(s).")

    # Draw rectangles around detected faces
    for top, right, bottom, left in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the image with detected faces
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

