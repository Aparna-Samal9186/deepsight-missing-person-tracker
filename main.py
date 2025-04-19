# main.py
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
import logging
from src.recognition.db_manager import convert_image_to_base64, store_embedding, verify_user, create_user, get_all_missing_persons
from src.detection.face_detector import detect_faces
from src.recognition.recognizer_utils import extract_embeddings
from src.recognition.identify_missing_person import identify_missing_person  # Import the new function
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("✅ Starting FastAPI...")
print(f"OpenCV version: {cv2.__version__}")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/test")
async def test():
    print("✅ /test endpoint hit!")
    return {"message": "API is working!"}

print("✅ main.py executed successfully!")

# Print all routes on startup
@app.on_event("startup")
async def startup_event():
    routes = [route.path for route in app.routes]
    logger.debug(f"Available routes: {routes}")

@app.post("/add_missing_person")
async def add_missing_person_api(
    image: UploadFile = File(...),
    missing_person_name: str = Form(...)
):
    print(f"Received image: {image.filename}, Name: {missing_person_name}")

    try:
        # Read and decode image file
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            print("❌ Image decoding failed. It might be corrupted or an unsupported format.")
            return {"error": "Invalid image format."}

        print(f"✅ Image shape: {image_np.shape}, dtype: {image_np.dtype}")

        # Ensure image is 3-channel (BGR)
        if len(image_np.shape) == 2:
            print("⚠️ Image is grayscale, converting to BGR")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] != 3:
            print("❌ Image does not have 3 channels (RGB/BGR).")
            return {"error": "Unsupported image type. Must be RGB or grayscale."}

        # Convert image to Base64 for storage
        image_base64 = convert_image_to_base64(image_np)

        # Detect faces and extract embeddings
        image_with_rects, faces = detect_faces(image_np.copy())  # Pass a copy
        if not faces:
            print("⚠️ No faces detected in the image.")
            return {"error": "No faces detected."}

        embeddings = []
        for face in faces:
            embedding = extract_embeddings(face)
            embeddings.append(embedding)

        # Store the first embedding (or all if needed)
        if embeddings:
            store_embedding(missing_person_name, embeddings[0], image_np, "missing_persons")
        else:
            print("⚠️ No embeddings extracted.")
            return {"error": "No valid face embeddings."}

        return {"message": f"{missing_person_name} added successfully!", "image_base64": image_base64}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": "Internal server error", "details": str(e)}
    

@app.post("/signup")
async def signup_api(
    username: str = Form(...),
    password: str = Form(...)
):
    result = create_user(username, password)
    return result

@app.post("/login")
async def login_api(
    username: str = Form(...),
    password: str = Form(...)
):
    result = verify_user(username, password)
    return result


@app.post("/identify_missing_person")
async def identify_missing_person_api(
    name: str = Form(...),
    age: int = Form(None),
    lost_location: str = Form(...),
    description: str = Form(None),
    contact: str = Form(...),
    image: UploadFile = File(...)
):
    print(f"Received report for: {name}, lost at: {lost_location}, Contact: {contact}, Image: {image.filename}")
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            return {"error": "Invalid image format."}

        # Now you have access to:
        # - name: str
        # - age: int (can be None)
        # - lost_location: str
        # - description: str (can be None)
        # - contact: str
        # - image: UploadFile (with image.filename, await image.read(), etc.)

        match_result = identify_missing_person(image_np, name=name, age=age, lost_location=lost_location, description=description, contact=contact) # Pass the additional data to your identification function
        return match_result

    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}

@app.get("/missing-persons")
def fetch_missing_persons():
    persons = get_all_missing_persons()
    return JSONResponse(content=persons)

@app.get("/")
async def root():
    logger.debug("Root endpoint hit!")
    return {"message": "Welcome to the DeepSight Missing Person Tracker API!"}