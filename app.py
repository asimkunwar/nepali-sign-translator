import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# --- CONFIGURATION ---
MODEL_FILE = "sanket_model.pkl"
TOTAL_EXPECTED = 84  # 2 hands x 21 joints x 2 coords
# ---------------------

# 1. Load trained model
print("🧠 Loading SanketAI Engine Brain...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print("✅ Model loaded successfully!")

# 2. Setup MediaPipe Dual Hand Tracking
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)

cap = cv2.VideoCapture(0)
print("🎥 Launching Realtime Nepali Sign Language Translator...")

HAND_COLORS = [(0, 255, 0), (0, 200, 255)]  # green, yellow

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        row_data = []
        prediction_text = "Show BOTH Hands"
        confidence_score = 0.0
        hands_detected = 0

        if detection_result and detection_result.hand_landmarks:
            hands_detected = len(detection_result.hand_landmarks)

            # FIX: hand is a list of landmarks, so use hand[0].x (wrist) not hand.x
            sorted_hands = sorted(
                detection_result.hand_landmarks,
                key=lambda hand: hand[0].x
            )

            for hand_index, hand_landmarks in enumerate(sorted_hands):
                color = HAND_COLORS[hand_index % len(HAND_COLORS)]
                for lm in hand_landmarks:
                    lx, ly = lm.x, lm.y
                    cx, cy = int(lx * w), int(ly * h)
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    row_data.append(float(lx))
                    row_data.append(float(ly))

            # --- REALTIME PREDICTION ---
            if hands_detected == 2 and len(row_data) == TOTAL_EXPECTED:
                input_features = np.array([row_data])

                # FIX: model.predict returns an array like ['Namaste'], extract [0]
                prediction_text = model.predict(input_features)[0].upper()

                probabilities = model.predict_proba(input_features)
                confidence_score = np.max(probabilities) * 100

        # --- UI OVERLAY ---
        # Top bar background
        cv2.rectangle(frame, (0, 0), (w, 75), (30, 30, 30), -1)

        # FIX: compare against the actual default string (not uppercased version)
        if prediction_text != "Show BOTH Hands".upper():
            cv2.putText(frame, f"TRANSLATION: {prediction_text}", (20, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(frame, f"Match: {confidence_score:.1f}%", (w - 200, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Show BOTH Hands", (20, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 2)

        # Hands detected counter
        cv2.putText(frame, f"Hands: {hands_detected}/2", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Footer
        cv2.putText(frame, "SanketAI v1.0 Live Interpreter | Press 'q' to Exit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('SanketAI - Nepali Sign Language Interpreter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()