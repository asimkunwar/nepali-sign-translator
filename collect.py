import cv2
import mediapipe as mp
import csv
import os
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# --- CONFIGURATION ---
CURRENT_SIGN = "Namaste"      # Change this for each sign!
CSV_FILE = "sign_dataset.csv"
TOTAL_EXPECTED = 84          # 2 hands × 21 landmarks × 2 coords (x, y)
SAMPLES_TO_COLLECT = 50       # Number of rows to collect automatically
RECORD_DELAY_MS = 200        # Time between auto-saves (200ms = 5 captures per second)
# ---------------------

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)

cap = cv2.VideoCapture(0)
print(f"🚀 Preparing Hands-Free Automation for Sign: '{CURRENT_SIGN}'")

# Tracking states
is_recording = False
samples_saved = 0
last_save_time = 0
countdown_start_time = None

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
        hands_detected = 0

        if detection_result and detection_result.hand_landmarks:
            hands_detected = len(detection_result.hand_landmarks)

            # FIX: hand is a list of 21 landmarks, so use hand[0].x (wrist) not hand.x
            sorted_hands = sorted(
                detection_result.hand_landmarks,
                key=lambda hand: hand[0].x
            )

            for hand_index, hand_landmarks in enumerate(sorted_hands):
                color = HAND_COLORS[hand_index % len(HAND_COLORS)]
                for lm in hand_landmarks:
                    lx, ly = lm.x, lm.y
                    cx, cy = int(lx * w), int(ly * h)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    row_data.append(float(lx))
                    row_data.append(float(ly))

        # --- AUTOMATION LOGIC ---
        current_time_ms = int(time.time() * 1000)
        hud_status = "Press 'R' to start countdown"
        status_color = (255, 255, 255)

        if not is_recording and countdown_start_time is not None:
            elapsed = time.time() - countdown_start_time
            remaining = 3 - int(elapsed)
            if remaining > 0:
                hud_status = f"GET READY! Starting in {remaining}s..."
                status_color = (0, 165, 255)  # Orange
            else:
                is_recording = True
                countdown_start_time = None
                samples_saved = 0
                last_save_time = current_time_ms

        elif is_recording:
            hud_status = f"RECORDING: {samples_saved}/{SAMPLES_TO_COLLECT}"
            status_color = (0, 0, 255)  # Red

            if hands_detected == 2 and len(row_data) == TOTAL_EXPECTED:
                if current_time_ms - last_save_time >= RECORD_DELAY_MS:
                    record_row = [CURRENT_SIGN] + row_data
                    file_exists = os.path.isfile(CSV_FILE)
                    with open(CSV_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            headers = (
                                ['label']
                                + [f'lh_j{i}_x' for i in range(21)]
                                + [f'lh_j{i}_y' for i in range(21)]
                                + [f'rh_j{i}_x' for i in range(21)]
                                + [f'rh_j{i}_y' for i in range(21)]
                            )
                            writer.writerow(headers)
                        writer.writerow(record_row)

                    samples_saved += 1
                    last_save_time = current_time_ms
                    print(f"✅ Auto-saved sample {samples_saved}/{SAMPLES_TO_COLLECT}")

            if samples_saved >= SAMPLES_TO_COLLECT:
                is_recording = False
                print(f"🎉 Successfully collected {SAMPLES_TO_COLLECT} samples for '{CURRENT_SIGN}'!")

        # --- ON-SCREEN HUD ---
        cv2.putText(frame, f"Sign Label: {CURRENT_SIGN}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, hud_status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Hands Tracked: {hands_detected}/2", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, "Controls: 'r' = Record (Timer) | 'q' = Quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('SanketAI - Hands-Free Auto Data Collector', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not is_recording:
            countdown_start_time = time.time()

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()