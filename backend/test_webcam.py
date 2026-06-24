import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup absolute paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "sign_language_model.pkl")
TASK_PATH = os.path.join(BACKEND_DIR, "hand_landmarker.task")
TOTAL_EXPECTED = 84

print("🔮 Loading trained AI brain and MediaPipe assets...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model not found at {MODEL_PATH}. Train it first!")
    exit()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# 2. Initialize MediaPipe Hand Detector
# Lowered confidence thresholds (default is 0.5) so the second hand is
# detected more reliably — especially when it's slightly angled, partially
# overlapping, or less well-lit than the first hand.
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Fire up the physical Webcam
cap = cv2.VideoCapture(0)

# Bump up resolution so smaller/farther hands have more detail for MediaPipe
# to work with. If your webcam doesn't support 1280x720 it will just fall
# back to its closest supported resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🎥 Webcam streaming activated! Press 'q' to exit the frame.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a natural selfie-view experience
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Track hands
    result = detector.detect(mp_image)

    prediction_text = "Waiting for hands..."
    coordinates = []
    hands_detected = len(result.hand_landmarks) if result.hand_landmarks else 0

    if result.hand_landmarks:
        # Sort hands left-to-right by wrist coordinate (index 0)
        sorted_hands = sorted(result.hand_landmarks, key=lambda hand: hand[0].x)

        for hand_landmarks in sorted_hands:
            for lm in hand_landmarks:
                # Draw visual knuckles directly onto your live camera screen
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                coordinates.extend([lm.x, lm.y])

        # If both hands are fully tracking, query the Random Forest brain
        if len(coordinates) == TOTAL_EXPECTED:
            features = np.array(coordinates).reshape(1, -1)
            prediction = model.predict(features)
            prediction_text = f"Predicted: {prediction}"
        else:
            prediction_text = f"Only {hands_detected} hand(s) detected - show BOTH"

    # Display HUD text overlay
    cv2.putText(frame, prediction_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Small debug readout so you can see detection count live while tuning
    cv2.putText(frame, f"Hands detected: {hands_detected}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

    # Show the camera interface window
    cv2.imshow("SanketAI - Live Sign Translation Matrix", frame)

    # Break loop cleanly when user hits 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam tracking terminated cleanly.")