# Smart Home Voice Control System

A web-based smart home controller integrating voice recognition and manual controls to manage devices like lights, fans, music, AC, and door locks.

## Overview
This project demonstrates a prototype smart home system built with Python, Flask, and machine learning. Users can issue voice commands (e.g., "play music") or use a web UI to control devices, with real-time feedback via colored toasts (green for success, red for errors) and logged actions.

## Features
- **Voice Control**: Uses Google Speech Recognition API for natural language input.
- **Web Interface**: Responsive GUI with buttons, device states, and logs.
- **Intent Recognition**: Naive Bayes classifier trained on 700+ augmented examples.
- **Real-Time Updates**: Device states refresh every 5 seconds.
- **Logging**: Actions saved to `device_logs.json`.

## Technologies
- **Backend**: Flask (`app.py`) on port 5000.
- **Voice Assistant**: Flask-SocketIO (`voice_assistant.py`) on port 5001.
- **NLP**: Scikit-learn (`train_intents.py`) with TF-IDF and MultinomialNB.
- **Frontend**: HTML/CSS/JS (`index.html`) with SocketIO.
- **APIs**: Local Flask API, Google Speech Recognition.

## Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/abhinavsathi/SmartHomeVoiceControl.git
   cd SmartHomeVoiceControl
2. Install Dependencies:
   ```bash
   pip install flask flask-socketio requests numpy scipy scikit-learn speechrecognition sounddevice matplotlib
3. Train Model
   ```bash
   python train_intents.py
4. Run Servers
   ```bash
   *Backend*:
    python app.py
   
   *Voice Assistant*:
   python voice_assistant.py
5. Access UI
   Open http://127.0.0.1:5000 in a browser. 

## Usage
- Manual: Click UI buttons (e.g., "Light On").
- Voice: Click the mic button, speak (e.g., "set AC to 16 degrees").
- Logs: View device_logs.json or UI logs section.
## Files
- 'app.py' : Flask server and API.
- 'voice_assistant.py' : Voice processing and SocketIO.
- 'train_intents.py' : Intent model training.
- 'intents.json': Training data (17 intents, 255+ patterns).
- 'templates/index.html': Web UI.
- 'device_logs.json, app.log, assistant.log': Logs.
- 'intent_model.pkl, vectorizer.pkl': Trained model files.
- 'confusion_matrix.png': Model performance visualization.
## Results
- Recognizes 17 intents with ~90% accuracy (cross-validation).
- Handles casual speech (e.g., "it’s too hot, can you increase the").
- Provides visual feedback and logs all actions.
## Future Enhancements
- Integrate external APIs (e.g., weather data).
- Expand intents (e.g., "open window").
- Deploy online with a public server.
