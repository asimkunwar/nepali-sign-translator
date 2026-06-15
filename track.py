import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup the modern MediaPipe Landmarker Configurations
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6
)

# Create the live landmark detector
detector = vision.HandLandmarker.create_from_options(options)

# Open the webcam feed
cap = cv2.VideoCapture(0)
print("Launching Webcam tracking with Tasks API... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera frame not detected.")
        continue

    # Mirror mirror on the wall
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert image format from OpenCV's BGR to MediaPipe's RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run the model frame detection
    detection_result = detector.detect(mp_image)

    # Custom Drawing Logic: If a hand landmark list is returned
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw the 21 joints onto your camera feed
            for landmark in hand_landmarks:
                # Convert normalized coordinates (0.0 - 1.0) to actual pixel dimensions
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) # Green dot

    # Show the output frame
    cv2.imshow('SanketAI - Modern Tracking Feed', frame)

    # Close when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely close down hardware resources
cap.release()
cv2.destroyAllWindows()