import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
filename = "hand_landmarker.task"

print("Downloading the model file... This might take a few seconds...")
urllib.request.urlretrieve(url, filename)

if os.path.exists(filename):
    print(f"🎉 Success! '{filename}' has been downloaded directly into your project folder.")
else:
    print("❌ Something went wrong with the download.")