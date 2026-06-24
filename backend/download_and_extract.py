import os
import csv
import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Target settings
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BACKEND_DIR, "extracted_sign_data.csv")
TOTAL_EXPECTED = 84  # Fixed size matrix (2 hands x 21 joints x 2 coordinates)

DATASET_FOLDERS = [
    os.path.join(BACKEND_DIR, "alphabet_dataset"),
    os.path.join(BACKEND_DIR, "word_dataset")
]

processed_labels = set()
file_exists = os.path.exists(OUTPUT_CSV)

if file_exists and os.path.getsize(OUTPUT_CSV) > 0:
    try:
        existing_df = pd.read_csv(OUTPUT_CSV, usecols=[0])
        processed_labels = set(existing_df.iloc[:, 0].dropna().astype(str).unique())
        print(f"✨ Found existing progress! Skipping {len(processed_labels)} labels.")
    except Exception:
        print("📝 CSV uninitialized. Starting fresh...")

# Initialize MediaPipe
model_path = os.path.join(BACKEND_DIR, "hand_landmarker.task")
if not os.path.exists(model_path):
    print("📥 Downloading hand_landmarker.task tracking asset...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Setup headers
with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists or os.path.getsize(OUTPUT_CSV) == 0:
        headers = ["label"] + [f"coord_{i}" for i in range(TOTAL_EXPECTED)]
        writer.writerow(headers)

print("🚀 Starting Smart Single & Double Hand Landmark Extraction Pipeline...")

for dataset_path in DATASET_FOLDERS:
    if not os.path.exists(dataset_path):
        continue

    print(f"\n📂 Processing Dataset: {os.path.basename(dataset_path)}")

    for root, dirs, files in os.walk(dataset_path):
        valid_images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if valid_images:
            category_name = os.path.basename(root).strip()

            if category_name in processed_labels:
                print(f"⏭️ Skipping label '{category_name}'")
                continue

            print(f"📸 Extracting signs for label: '{category_name}'")
            frame_count = 0

            for img_name in valid_images:
                img_path = os.path.join(root, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect(mp_image)

                coordinates = []
                if result.hand_landmarks:
                    # Sort left to right using each hand's wrist landmark (index 0)
                    sorted_hands = sorted(result.hand_landmarks, key=lambda hand: hand[0].x)

                    for hand_landmarks in sorted_hands:
                        for lm in hand_landmarks:
                            coordinates.extend([lm.x, lm.y])

                # ⚡ OPTION A FIX: DYNAMIC ZERO PADDING LAYER ⚡
                # If only 1 hand (42 coordinates) is found, pad the rest with 0.0 up to 84
                if 0 < len(coordinates) < TOTAL_EXPECTED:
                    while len(coordinates) < TOTAL_EXPECTED:
                        coordinates.append(0.0)

                # Now both 1-hand and 2-hand signs securely save!
                if len(coordinates) == TOTAL_EXPECTED:
                    with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([category_name] + coordinates)
                    frame_count += 1

            print(f"✅ Successfully added {frame_count} rows for label '{category_name}'")

print(f"\n🎉 DATASET EXTRACTION COMPLETE! Saved at: {OUTPUT_CSV}")