import os
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Setup absolute paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BACKEND_DIR, "extracted_sign_data.csv")
MODEL_OUTPUT = os.path.join(BACKEND_DIR, "sign_language_model.pkl")

print("🧠 Loading extracted landmark dataset...")
if not os.path.exists(DATA_CSV):
    print(f"❌ Error: Could not find {DATA_CSV}. Wait for the extractor script to finish!")
    exit()

# 2. Read CSV Data & Apply Type Fixes
# low_memory=False stops pandas from guessing mixed types per chunk
df = pd.read_csv(DATA_CSV, low_memory=False)

# Normalize the label column: force to string, strip whitespace.
# This keeps numeric-looking labels (e.g. "5", "10") and text labels
# (e.g. "नमस्कार") consistent, so "5" and " 5 " aren't treated as different classes.
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

# Drop any accidental fully-empty rows
df = df.dropna()

print(f"📊 Dataset Loaded! Total data samples collected: {df.shape}")

# 3. Split into features (X) and labels (y)
X = df.iloc[:, 1:].values  # Columns 1 to 84 (the coordinates)
y = df.iloc[:, 0].values   # Column 0 (the sign label names)

# Check if we actually have data
if len(X) == 0:
    print("❌ The CSV file is empty. MediaPipe didn't find hands in the images.")
    exit()

# 3b. 🧹 FILTER OUT CLASSES WITH TOO FEW SAMPLES
# A stratified split needs >=2 samples per class. Build a boolean occurrence
# bitmask over y so any label appearing fewer than 2 times gets dropped.
label_counts = pd.Series(y).value_counts()
rare_labels = label_counts[label_counts < 2].index
keep_mask = ~pd.Series(y).isin(rare_labels).values

if (~keep_mask).any():
    dropped = label_counts[label_counts < 2].index.tolist()
    print(f"\n⚠️ Dropping {len(dropped)} label(s) with fewer than 2 samples: {dropped}")
    print("   (Add more training images for these labels and re-run the extractor.)")
    X = X[keep_mask]
    y = y[keep_mask]
    print(f"📊 Dataset after filtering: {X.shape[0]} samples across {len(set(y))} labels")

# 4. Train/Test Split (80% training, 20% validation testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔄 Training on {len(X_train)} frames, validating on {len(X_test)} frames...")

# 5. Initialize and Train a fast Linear SVM
# - StandardScaler: LinearSVC is sensitive to feature scale; landmark x/y
#   coordinates are already 0-1 normalized by MediaPipe, but scaling still
#   helps the solver converge faster and more reliably.
# - LinearSVC(dual=False): the primal formulation is the right choice when
#   n_samples > n_features (84 features, thousands of samples here), and is
#   noticeably faster to train and to predict with than the dual form.
# - No CalibratedClassifierCV: calibration adds a real training-time and
#   inference-time cost for probability outputs we don't strictly need.
#   We use the raw decision_function margin as a fast confidence proxy
#   instead (see backend/main.py for how this is converted to a 0-100 score).
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(dual=False, C=1.0, max_iter=5000, random_state=42))
])

train_start = time.time()
model.fit(X_train, y_train)
print(f"⏱️ Training completed in {time.time() - train_start:.2f}s")

# 6. Evaluate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")

print("\n📝 Detailed Performance Report:")
print(classification_report(y_test, y_pred))

# 6b. Sanity-check single-frame inference speed (the actual production path)
sample = X_test[:1]
timings = []
for _ in range(200):
    t0 = time.perf_counter()
    _ = model.decision_function(sample)
    timings.append((time.perf_counter() - t0) * 1000)
print(f"⚡ Avg single-frame inference time: {np.mean(timings):.3f} ms "
      f"(max {np.max(timings):.3f} ms over 200 runs)")

# 7. Save the trained model to a file
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(model, f)

print(f"\n🎉 SUCCESS! Your trained AI brain is saved at: {MODEL_OUTPUT}")
print("You can now restart your FastAPI backend to use the updated dataset dictionary!")