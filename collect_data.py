import cv2
import mediapipe as mp
import csv
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CURRENT_SIGN = "Namaste" 
CSV_FILE = "sign_dataset.csv"
# ---------------------

# Setup MediaPipe Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, 
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# Open Webcam
cap = cv2.VideoCapture(0)
print(f"Data Collection Started for Sign: '{CURRENT_SIGN}'")
print("Instructions: Make the sign, then press 's' to SAVE. Press 'q' to QUIT.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Temporary list to hold coordinates for this frame
    row_data = []

    # Safe checking of nested Tasks API structure
    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
        single_hand_points = detection_result.hand_landmarks 
        
        for landmark in single_hand_points:
            # Calculate pixel positions for display
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Save raw coordinates (unpacked correctly here)
            row_data.append(landmark.x)
            row_data.append(landmark.y)

    # Show instructions on the interface window
    cv2.putText(frame, f"Recording: {CURRENT_SIGN}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Press 's' to Save | 'q' to Quit", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('SanketAI - Data Collector', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Save values to spreadsheet when 's' key is pressed
    if key == ord('s'):
        if len(row_data) == 42: # 21 points * 2 dimensions (x, y)
            row_data.insert(0, CURRENT_SIGN)
            
            file_exists = os.path.isfile(CSV_FILE)
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    headers = ['label'] + [f'j{i}_x' for i in range(21)] + [f'j{i}_y' for i in range(21)]
                    writer.writerow(headers)
                
                writer.writerow(row_data)
            print(f"✅ Saved 1 data point for '{CURRENT_SIGN}'")
        else:
            print("❌ Show your hand completely to the camera before pressing 's'!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()