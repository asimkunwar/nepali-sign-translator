import pickle
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your frontend website to talk to your backend server safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🇳🇵 Translation Dictionary
NEPALI_DICTIONARY = {
    "Namaste": "नमस्ते",
    "Ka": "क",
    "Kha": "ख",
    "Amma": "आमा",
    "Idle": "सामान्य अवस्था"
}

# Load your custom AI model brain
MODEL_PATH = "backend/sanket_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("🧠 SanketAI Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Could not load model file at {MODEL_PATH}: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "SanketAI Backend Server Running"}

# ⚡ Live High-Speed WebSockets Endpoint for Real-time Translation
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎥 Live webcam stream connected via WebSocket!")
    
    try:
        while True:
            # Receive coordinates array from the frontend browser
            data = await websocket.receive_json()
            coordinates = data.get("coordinates", [])
            
            # If we get exactly 84 coordinates (2 hands tracked)
            if len(coordinates) == 84 and model is not None:
                input_features = np.array([coordinates])
                prediction = model.predict(input_features)
                
                # Get confidence score
                probabilities = model.predict_proba(input_features)
                confidence = float(np.max(probabilities) * 100)
                
                english_label = str(prediction)
                nepali_text = NEPALI_DICTIONARY.get(english_label, english_label)
                
                # Stream the answer straight back to the website instantly
                await websocket.send_json({
                    "nepali_text": nepali_text,
                    "confidence": f"{confidence:.1f}%"
                })
            else:
                await websocket.send_json({
                    "nepali_text": "दुवै हात देखाउनुहोस् (Show Both Hands)",
                    "confidence": "0%"
                })
                
    except WebSocketDisconnect:
        print("🎥 Webcam stream disconnected.")