import cv2
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
from datetime import datetime, timedelta
import time

# Initialize Mediapipe and CVZone
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
detector = FaceMeshDetector(maxFaces=1)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Variables to track detection status
face_detected = False
start_time = None
no_face_start_time = None

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Convert the image to RGB for Mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if results.detections:
        if not face_detected:
            print("Face Detected")
            print(datetime.now())
            face_detected = True
            
        # Draw bounding box around detected face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            
            # Use CVZone to find facial landmarks
            img, faces = detector.findFaceMesh(img)
    
    else:
        if face_detected:
            print("No Face Detected")
            face_detected = False
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
