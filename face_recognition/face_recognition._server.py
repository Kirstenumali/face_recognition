import cv2
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
from datetime import datetime, timedelta
import json
import os

# Initialize Mediapipe and CVZone
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
detector = FaceMeshDetector(maxFaces=1)

# File to store timestamps and greeting status
data_file = 'data.json'

# Load existing data or initialize it
if os.path.exists(data_file):
    with open(data_file, 'r') as file:
        data = json.load(file)
else:
    data = {
        "face_detection_start_time": None,
        "greeting_triggered": False
    }

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Convert the image to RGB for Mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if results.detections:
        if data["face_detection_start_time"] is None:
            data["face_detection_start_time"] = datetime.now().isoformat()
            data["greeting_triggered"] = False
            print("Face Detected at:", data["face_detection_start_time"])
            # Save the data to the JSON file
            with open(data_file, 'w') as file:
                json.dump(data, file)
        
        # Calculate elapsed time since face detection started
        face_detection_start_time = datetime.fromisoformat(data["face_detection_start_time"])
        elapsed_time = datetime.now() - face_detection_start_time
        print(f"Face detected for: {elapsed_time.seconds} seconds", end="\r")

        # Trigger greeting after 1.5 minutes (90 seconds)
        if elapsed_time >= timedelta(minutes=1.5) and not data["greeting_triggered"]:
            print("\nGreeting: Hello! You've been detected for 1.5 minutes!")
            data["greeting_triggered"] = True
            # Save the updated data to the JSON file
            with open(data_file, 'w') as file:
                json.dump(data, file)
        
        # Draw bounding box around detected face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            
            # Use CVZone to find facial landmarks
            img, faces = detector.findFaceMesh(img)
    
    else:
        if data["face_detection_start_time"] is not None:
            # Reset face detection status when face is no longer detected
            data["face_detection_start_time"] = None
            data["greeting_triggered"] = False
            print("\nNo Face Detected")
            # Save the reset data to the JSON file
            with open(data_file, 'w') as file:
                json.dump(data, file)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
