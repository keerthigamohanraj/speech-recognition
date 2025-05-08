import os
import time
import pickle
import numpy as np
import librosa

# Load your trained model
with open('speech_command_model.pkl', 'rb') as file:
    import joblib
model = joblib.load('speech_command_model.pkl')


# Function to extract MFCC from audio file
def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Initialize light variables
light_running = False
start_time = 0

while True:
    audio_path = input("Enter WAV filename (or 'exit' to quit): ")

    if audio_path.lower() == 'exit':
        print("Exiting light.")
        break

    if not os.path.exists(audio_path):
        print("File not found. Try again.")
        continue

    try:
        # Extract features from the audio file
        features = extract_mfcc(audio_path).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)[0]
        print(f"Predicted Command: {prediction}")

        # Handle light based on prediction
        if prediction == 'on':
            if not light_running:
                light_running = True
                start_time = time.time()
                print("light on. Speak 'off' to end the light.")
            else:
                print("light already running.")
        elif prediction == 'off':
            if light_running:
                elapsed = time.time() - start_time
                light_running = False
                print(f"light off. you used light  {round(elapsed, 2)} seconds.")
            else:
                print("light is not running yet.")
        else:
            print("Unknown command. Only 'on' and 'off' are supported.")
    except Exception as e:
        print("Error processing file:", str(e))
