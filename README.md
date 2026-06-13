# nepali-sign-translator
This is the system that will be used to translate the sign language into nepali language.

# SanketAI a Nepali Sign Language Translation System
 An intelligent computer vision system designed to recognize and translate Nepali Sign Language (NSL) gestures into text and spoken Nepali language. This project aims to bridge the communication gap between the deaf/hard-of-hearing community in Nepal and the general public.

# Project Overview
Most existing sign language recognition systems focus heavily on American Sign Language (ASL) or English translations. This system focuses specifically on Nepali Sign Language (NSL), localizing the solution to recognize unique cultural gestures, alphabets (Devanagari script), and common words used in Nepal.

# Tech Stack & Frameworks

    1. Language: Python

    2. Computer Vision: OpenCV (Video capture & frame processing)

    3. Feature Extraction: Google MediaPipe (Hand landmark tracking)

    4. Machine Learning: Scikit-learn / TensorFlow (Gesture classification)

    5. Interface: Streamlit (For a quick, clean web application)

# Project Roadmap

    To keep development organized, the project is divided into three key milestones:

    🔹 Phase 1: Static Fingerspelling Recognition (Current Focus)Build a custom dataset of Nepali alphabets (क, ख, ग...) and numbers.Use MediaPipe to extract $(x, y, z)$ coordinates of hand landmarks.Train a lightweight machine learning model (e.g., Random Forest or SVM) to output Devanagari text.
    🔹 Phase 2: Dynamic Word RecognitionExtend the system to recognize motion-based gestures (e.g., "Namaste", "Ghar", "Khana").Implement sequence models like LSTM (Long Short-Term Memory) to process multi-frame actions.
    🔹 Phase 3: Text-to-Speech IntegrationConvert the recognized Nepali text into spoken audio using Python text-to-speech engines.