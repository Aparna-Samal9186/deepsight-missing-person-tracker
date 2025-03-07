import dlib

predictor_path = r"C:\Users\APARNA SAMAL\Desktop\DeepSight\deepsight-missing-person-tracker\models\shape_predictor_68_face_landmarks.dat"

try:
    predictor = dlib.shape_predictor(predictor_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")