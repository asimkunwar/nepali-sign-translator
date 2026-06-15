import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset we created
CSV_FILE = "sign_dataset.csv"
print("📊 Loading dataset...")
df = pd.read_csv(CSV_FILE)

print(f"   Total samples: {len(df)}")
print(f"   Signs found:   {df['label'].unique().tolist()}")

# Separate features (the 84 coordinates) and labels (the sign names)
X = df.drop(columns=['label'])
y = df['label']

# 2. Split data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # FIX: was test_split → test_size
)

print(f"📈 Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# 3. Initialize and train the Random Forest AI model
print("🤖 Training the AI model (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. Evaluate how accurate the model is
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Training Complete! Accuracy Score: {accuracy * 100:.2f}%")

# Per-sign breakdown so you can see which signs need more data
print("\n📋 Per-Sign Report:")
print(classification_report(y_test, y_pred))

# 5. Save the trained model so the webcam script can use it
MODEL_FILE = "sanket_model.pkl"
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Saved AI model as '{MODEL_FILE}' — ready for live detection!")