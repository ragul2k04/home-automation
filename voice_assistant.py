import speech_recognition as sr
import sounddevice as sd
import numpy as np
import requests
import json
import os
import pickle
from datetime import datetime
import logging
from flask import Flask
from flask_socketio import SocketIO
import re
from google.cloud import speech
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='assistant.log'
)
logger = logging.getLogger('voice_assistant')

# API endpoint
API_ENDPOINT = "http://127.0.0.1:5000/command"
LOG_FILE = "assistant_logs.json"
MODEL_FILE = "intent_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"


def normalize_intent_for_api(intent_or_text):
    """
    Ensure the command sent to the Flask API uses canonical intent names like 'turn_on_light'.
    Accepts either model intent names or short/natural text variants and returns canonical intent.
    """
    if not intent_or_text:
        return intent_or_text
    s = intent_or_text.strip().lower()
    # if already in intent form (has underscore), assume it's canonical
    if "_" in s and all(part.isalpha() or part.isdigit() for part in s.replace('_','').split()):
        return s
    # common mappings
    maps = {
        "off light": "turn_off_light",
        "on light": "turn_on_light",
        "turn on light": "turn_on_light",
        "turn off light": "turn_off_light",
        "switch on light": "turn_on_light",
        "switch off light": "turn_off_light",
        "lights on": "turn_on_light",
        "lights off": "turn_off_light",
        "on fan": "turn_on_fan",
        "off fan": "turn_off_fan",
        "turn on fan": "turn_on_fan",
        "turn off fan": "turn_off_fan",
        "play music": "play_music",
        "stop music": "stop_music",
        "lock door": "lock_front_door",
        "unlock door": "unlock_front_door"
    }
    if s in maps:
        return maps[s]
    # handle phrases containing keywords
    if "turn on" in s and "fan" in s:
        return "turn_on_fan"
    if "turn off" in s and "fan" in s:
        return "turn_off_fan"
    if "turn on" in s and "light" in s:
        return "turn_on_light"
    if "turn off" in s and "light" in s:
        return "turn_off_light"
    # fallback: replace spaces with underscores
    return s.replace(" ", "_")

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load trained AI model and vectorizer
def load_model():
    try:
        with open(MODEL_FILE, "rb") as model_file, open(VECTORIZER_FILE, "rb") as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}. Run train_intents.py first.")
        exit(1)

model, vectorizer = load_model()

# Log assistant usage
def log_assistant_usage(command, predicted_intent, status):
    log_entry = {
        "command": command,
        "predicted_intent": predicted_intent,
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if os.path.exists(LOG_FILE):
        logs = json.load(open(LOG_FILE, "r")) if os.path.exists(LOG_FILE) else []
    else:
        logs = []
    logs.append(log_entry)
    with open(LOG_FILE, "w") as file:
        json.dump(logs, file, indent=4)

# Record audio
def record_audio(duration=3, fs=44100):
    print("Listening... 🎙️")
    try:
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete! ✅")
        return np.frombuffer(audio_data, dtype=np.int16), fs
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        print(f"Error recording audio: {e}")
        return None, None

# Recognize speech
# def recognize_speech(audio_data_np, fs):
#     if audio_data_np is None or fs is None:
#         return None
#     recognizer = sr.Recognizer()
#     audio = sr.AudioData(audio_data_np.tobytes(), fs, 2)
#     try:
#         command = recognizer.recognize_google(audio)
#         print(f"You said: {command}")
#         return command.lower()
#     except sr.UnknownValueError:
#         print("Sorry, I couldn't understand that. 🙁")
#         return None
#     except sr.RequestError as e:
#         logger.error(f"Google API error: {e}")
#         print("Couldn't request results, check your internet connection. 📡")
#         return None
def recognize_speech(audio_data_np, fs):
    if audio_data_np is None or fs is None:
        return None
    
    try:
        # For standard recognizer fallback
        recognizer = sr.Recognizer()
        audio = sr.AudioData(audio_data_np.tobytes(), fs, 2)
        
        # Try Google Cloud Speech-to-Text first
        try:
            # Create Google client
            client = speech.SpeechClient()
            
            # Convert audio to proper format
            content = audio.get_wav_data()
            audio_google = speech.RecognitionAudio(content=content)
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=fs,
                language_code="en-US",
                model="command_and_search"
            )
            
            response = client.recognize(config=config, audio=audio_google)
            
            for result in response.results:
                command = result.alternatives[0].transcript
                print(f"You said (Google Cloud): {command}")
                logger.info(f"Google Cloud STT: {command}")
                return command.lower()
            
            # If no result from Google Cloud, fall back to standard recognizer
            raise Exception("No result from Google Cloud STT")
            
        except Exception as e:
            logger.warning(f"Google Cloud STT failed, falling back: {e}")
            command = recognizer.recognize_google(audio)
            print(f"You said (Fallback): {command}")
            return command.lower()
            
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that. 🙁")
        return None
    except sr.RequestError as e:
        logger.error(f"API error: {e}")
        print("Couldn't request results, check your internet connection. 📡")
        return None
# Extract temperature
def extract_temperature(command):
    match = re.search(r'(\d+)(?:\s*degrees|\s*celsius)?', command)
    if match:
        temp = int(match.group(1))
        if 16 <= temp <= 30:
            return temp
    return None

# Extract song name
def extract_song_name(command):
    # Relaxed regex to catch "play song X", "play the song X", etc.
    match = re.search(r'play\s*(?:the\s*)?song\s+(.+)', command, re.IGNORECASE)
    if match:
        song_name = match.group(1).strip()
        logger.info(f"Extracted song_name: {song_name}")
        return song_name
    logger.warning(f"No song name extracted from: {command}")
    return None

# Predict intent
def predict_intent(command):
    try:
        X_input = vectorizer.transform([command])
        predicted_intent = model.predict(X_input)[0]
        confidence = np.max(model.predict_proba(X_input))
        logger.info(f"Predicted intent: {predicted_intent} (confidence: {confidence:.2f})")
        if confidence < 0.4:
            logger.warning(f"Low confidence prediction: {predicted_intent} ({confidence:.2f})")
            return "unknown", {}
        
        params = {}
        if predicted_intent == "set_ac_temperature":
            temp = extract_temperature(command)
            if temp:
                params["temperature"] = temp
        elif predicted_intent == "play_music":
            song_name = extract_song_name(command)
            if song_name:
                params["song_name"] = song_name
                
        return predicted_intent, params
    except Exception as e:
        logger.error(f"Error predicting intent: {e}")
        return "unknown", {}

# Send command to API
def send_command_to_home(command, params=None):
    payload = {"command": normalize_intent_for_api(command)}
    if params:
        payload["params"] = params
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        if response.status_code == 200:
            print("Command executed successfully! 🎉")
            log_assistant_usage(command, command, "success")
            response_data = response.json()
            if "message" in response_data:
                print(response_data["message"])
            return True
        else:
            print(f"Failed to execute the command. Status: {response.status_code} ❌")
            log_assistant_usage(command, command, "failed")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
        print(f"Error: {e}")
        log_assistant_usage(command, command, "error")
        return False

# SocketIO event handler
@socketio.on('listen_now')
def listen_once():
    audio_data_np, fs = record_audio()
    command = recognize_speech(audio_data_np, fs)
    if command:
        intent, params = predict_intent(command)
        if intent == "set_ac_temperature":
            temp = params.get("temperature")
            if temp:
                success = send_command_to_home(intent, params)
                socketio.emit('command_result', {"message": f"Set temp to {temp}°C", "status": "success" if success else "error"})
            else:
                socketio.emit('command_result', {"message": "Specify a temp (16-30°C)", "status": "error"})
        elif intent in ["greeting", "farewell", "exit", "status_check", "help"]:
            messages = {
                "greeting": "Hello! How can I assist you?",
                "farewell": "Goodbye! See you soon.",
                "exit": "Shutting down... Bye!",
                "status_check": "Checking status... All systems nominal!",
                "help": "I can turn lights on/off, play music, lock doors, and more. Try 'play the song Sunflower'!"
            }
            socketio.emit('command_result', {"message": messages.get(intent, "Command executed"), "status": "success"})
        elif intent != "unknown":
            success = send_command_to_home(intent, params)
            socketio.emit('command_result', {"message": f"Executed {intent}", "status": "success" if success else "error"})
        else:
            socketio.emit('command_result', {"message": "Sorry, I didn’t get that", "status": "error"})

if __name__ == "__main__":
    print("Starting Voice Assistant with SocketIO on port 5001...")
    socketio.run(app, host='0.0.0.0', port=5001)