from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
import logging
import spotipy
from google import genai
from spotipy.oauth2 import SpotifyOAuth
import re

# Add these imports at the top with other imports
import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger('app')

app = Flask(__name__)
# Gemini API Setup
# Gemini API Setup
GEMINI_API_KEY = "AIzaSyCjxSmEZwemtjMuTAxYmKnifR8Wktrfz8U"
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config={
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 1024,
    },
)
# Path to the log file
LOG_FILE = "device_logs.json"

# Device state storage
DEVICE_STATE = {
    "light": "off",
    "fan": "off",
    "music": "stopped",
    "front_door": "locked",
    "ac_temp": 22
}

# Spotify setup
SPOTIFY_CLIENT_ID = "b6e627d8f4224a968b66b3e6c8c65da9"  # Your Client ID
SPOTIFY_CLIENT_SECRET = "fe23d961597c44249dfd675046319f5e"  # Your Client Secret
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:5000/callback"
SPOTIFY_SCOPE = "user-read-playback-state user-modify-playback-state"
sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SPOTIFY_SCOPE,
    cache_path=".spotifycache"
)
# @app.route("/chat", methods=["POST"])
# def handle_chat():
#     try:
#         data = request.get_json()
#         message = data.get("message", "")
        
#         if not message:
#             return jsonify({"status": "error", "message": "No message provided"}), 400
            
#         # Get device state to provide context
#         context = f"Smart home current status: Lights {DEVICE_STATE['light']}, " \
#                   f"Fan {DEVICE_STATE['fan']}, Music {DEVICE_STATE['music']}, " \
#                   f"Front door {DEVICE_STATE['front_door']}, Temperature {DEVICE_STATE['ac_temp']}°C."
                  
#         try:
#             response = model.generate_content([context, message])
#             return jsonify({"status": "success", "message": response.text})
#         except InvalidArgument as e:
#             logger.error(f"Gemini API error: {str(e)}")
#             return jsonify({"status": "error", "message": "Could not process your request"}), 500
            
#     except Exception as e:
#         logger.error(f"Error in chat handler: {str(e)}")
#         return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '').lower()

        # Context prompt for Gemini
        prompt = f"""
        You are a smart home assistant. Interpret user commands related to devices.
        Current device states:
        {DEVICE_STATE}

        User said: "{user_input}"
        Respond naturally and clearly what action you're taking.
        """

        response = model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}]
        )

        reply = response.text.strip() if hasattr(response, "text") else "I'm not sure how to respond."

        # 🔍 Predict and handle actual device command
        intent, params = predict_intent(user_input)

        if intent != "unknown":
            update_device_state(intent, params)  # ✅ Update backend state

            # Log change
            log_entry = {
                "command": intent,
                "params": params,
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            logs = load_logs()
            logs.append(log_entry)
            with open(LOG_FILE, "w") as f:
                json.dump(logs, f, indent=4)

            # Add real confirmation to Gemini reply
            confirmation = get_dynamic_response(intent, params)
            reply = f"{reply} {confirmation}"

        # ✅ Send current device states along with chat reply
        return jsonify({
            "status": "success",
            "message": reply,
            "devices": DEVICE_STATE
        })

    except Exception as e:
        print("❌ Error in /chat route:", e)
        return jsonify({"status": "error", "message": "Server error occurred."})
def get_spotify_client():
    try:
        token_info = sp_oauth.get_cached_token()
        if not token_info:
            auth_url = sp_oauth.get_authorize_url()
            logger.info(f"Open this URL in your browser: {auth_url}")
            print(f"Open this URL in your browser: {auth_url}")
            return None
        return spotipy.Spotify(auth=token_info['access_token'])
    except Exception as e:
        logger.error(f"Spotify token error: {str(e)}")
        return None

# Load device logs for UI
def load_logs():
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as file:
                logs = json.load(file)
            return logs
        return []
    except Exception as e:
        logger.error(f"Error loading logs: {str(e)}")
        return []

# UI for controlling devices
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint to get device logs
@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        logs = load_logs()
        return jsonify({"logs": logs, "devices": DEVICE_STATE})
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Dynamic responses for commands
def get_dynamic_response(command, params=None):
    responses = {
        "turn_on_light": "💡 Lights turned ON. Let there be light!",
        "turn_off_light": "🌑 Lights turned OFF. Going dark...",
        "turn_on_fan": "🌀 Fan is spinning! Cooling things down...",
        "turn_off_fan": "❄️ Fan turned OFF. Calm and quiet now.",
        "play_music": f"🎵 Playing '{params.get('song_name', 'music')}' on Spotify!",
        "stop_music": "🔇 Music stopped. Silence restored.",
        "lock_front_door": "🔒 Front door locked. Home secured.",
        "unlock_front_door": "🔓 Front door unlocked. Be careful!",
        "activate_movie_mode": "🎬 Movie mode activated! Grab some popcorn!",
        "activate_night_mode": "🌙 Night mode activated. Sleep tight!",
        "activate_away_mode": "🏡 Away mode activated. Be safe!",
        "set_ac_temperature": "🌡️ Temperature set successfully! Adjusting climate control..."
    }
    response = responses.get(command, "✅ Command executed successfully!")
    if command == "set_ac_temperature" and params and "temperature" in params:
        response = f"🌡️ Temperature set to {params['temperature']}°C. Adjusting climate control..."
    return response

# Update device state based on command
def update_device_state(command, params=None):
    if command == "turn_on_light":
        DEVICE_STATE["light"] = "on"
    elif command == "turn_off_light":
        DEVICE_STATE["light"] = "off"
    elif command == "turn_on_fan":
        DEVICE_STATE["fan"] = "on"
    elif command == "turn_off_fan":
        DEVICE_STATE["fan"] = "off"
    elif command == "play_music":
        DEVICE_STATE["music"] = "playing"
    elif command == "stop_music":
        DEVICE_STATE["music"] = "stopped"
    elif command == "lock_front_door":
        DEVICE_STATE["front_door"] = "locked"
    elif command == "unlock_front_door":
        DEVICE_STATE["front_door"] = "unlocked"
    elif command == "set_ac_temperature" and params and "temperature" in params:
        DEVICE_STATE["ac_temp"] = params["temperature"]
    elif command == "activate_movie_mode":
        DEVICE_STATE.update({"light": "off", "fan": "on", "music": "playing"})
    elif command == "activate_night_mode":
        DEVICE_STATE.update({"light": "off", "fan": "off", "front_door": "locked"})
    elif command == "activate_away_mode":
        DEVICE_STATE.update({"light": "off", "fan": "off", "front_door": "locked", "music": "stopped"})

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
    match = re.search(r'play\s*(?:the\s*)?song\s+(.+)', command, re.IGNORECASE)
    if match:
        song_name = match.group(1).strip()
        logger.info(f"Extracted song_name: {song_name}")
        return song_name
    logger.warning(f"No song name extracted from: {command}")
    return None

# Predict intent (temporary local implementation)
def predict_intent(command):
    # Simplified intent prediction (replace with actual model if needed)
    intents = ["turn_on_light", "turn_off_light", "turn_on_fan", "turn_off_fan", "play_music", 
               "stop_music", "lock_front_door", "unlock_front_door", "set_ac_temperature", 
               "activate_movie_mode", "activate_night_mode", "activate_away_mode", "unknown"]
    params = {}
    if "turn on light" in command:
        return "turn_on_light", params
    elif "turn off light" in command:
        return "turn_off_light", params
    elif "turn on fan" in command:
        return "turn_on_fan", params
    elif "turn off fan" in command:
        return "turn_off_fan", params
    elif "play" in command:
        song_name = extract_song_name(command)
        if song_name:
            params["song_name"] = song_name
        return "play_music", params
    elif "stop music" in command:
        return "stop_music", params
    elif "lock front door" in command:
        return "lock_front_door", params
    elif "unlock front door" in command:
        return "unlock_front_door", params
    elif "set ac temperature" in command:
        temp = extract_temperature(command)
        if temp:
            params["temperature"] = temp
        return "set_ac_temperature", params
    elif "activate movie mode" in command:
        return "activate_movie_mode", params
    elif "activate night mode" in command:
        return "activate_night_mode", params
    elif "activate away mode" in command:
        return "activate_away_mode", params
    return "unknown", {}

# Endpoint to control devices
@app.route("/command", methods=["POST"])
def handle_command():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON data."}), 400
            
        command = data.get("command")
        params = data.get("params", {})
        
        if not command:
            return jsonify({"status": "error", "message": "No command received."}), 400
        
        # Handle Spotify for music commands
        # if command == "play_music":
        #     sp = get_spotify_client()
        #     if sp:
        #         try:
        #             song_name = params.get("song_name")
        #             logger.info(f"Received song_name: {song_name}")
        #             if song_name:
        #                 results = sp.search(q=song_name, type="track", limit=1)
        #                 tracks = results["tracks"]["items"]
        #                 if tracks:
        #                     track_uri = tracks[0]["uri"]
        #                     track_name = tracks[0]["name"]
        #                     logger.info(f"Playing track: {track_name}, URI: {track_uri}")
        #                     sp.start_playback(uris=[track_uri])
        #                     update_device_state(command, params)
        #                     response_text = get_dynamic_response(command, params)
        #                 else:
        #                     logger.warning(f"Song '{song_name}' not found")
        #                     sp.start_playback(uris=["spotify:track:4cOdK2wGLETKBW3PvgPWqT"])  # Yesterday
        #                     update_device_state(command, params)
        #                     response_text = f"🎵 Song '{song_name}' not found, playing default track!"
        #             else:
        #                 logger.info("No song_name, playing default")
        #                 sp.start_playback(uris=["spotify:track:4cOdK2wGLETKBW3PvgPWqT"])  # Yesterday
        #                 update_device_state(command, params)
        #                 response_text = get_dynamic_response(command, params)
        #         except Exception as e:
        #             logger.error(f"Spotify error: {str(e)}")
        #             return jsonify({"status": "error", "message": f"Spotify error: {str(e)}"}), 500
        #     else:
        #         return jsonify({"status": "error", "message": "Spotify not authenticated. Check terminal."}), 401
        # Handle Spotify for music commands
        if command == "play_music":
            sp = get_spotify_client()
            if sp:
                try:
                    song_name = params.get("song_name")
                    logger.info(f"Received song_name: {song_name}")
                    if song_name:
                        results = sp.search(q=song_name, type="track", limit=5)
                        tracks = results["tracks"]["items"]
                        if tracks:
                            track_uri = tracks[0]["uri"]
                            track_name = tracks[0]["name"]
                            artist_name = tracks[0]["artists"][0]["name"]
                            logger.info(f"Playing track: {track_name} by {artist_name}, URI: {track_uri}")
                            sp.start_playback(uris=[track_uri])
                            update_device_state(command, params)
                            response_text = f"🎵 Playing '{track_name}' by {artist_name}!"
                        else:
                            logger.warning(f"Song '{song_name}' not found")
                            response_text = f"🎵 Sorry, I couldn't find '{song_name}'. Please try another song."
                    else:
                        logger.info("No song_name, showing song selector")
                        update_device_state(command, params)
                        response_text = "🎵 Please select a song from the music player"
                except Exception as e:
                    logger.error(f"Spotify error: {str(e)}")
                    return jsonify({"status": "error", "message": f"Spotify error: {str(e)}"}), 500
            else:
                return jsonify({"status": "error", "message": "Spotify not authenticated. Check terminal."}), 401
        elif command == "stop_music":
            sp = get_spotify_client()
            if sp:
                try:
                    sp.pause_playback()
                    update_device_state(command, params)
                    response_text = get_dynamic_response(command, params)
                except Exception as e:
                    logger.error(f"Spotify error: {str(e)}")
                    return jsonify({"status": "error", "message": f"Spotify error: {str(e)}"}), 500
            else:
                return jsonify({"status": "error", "message": "Spotify not authenticated. Check terminal."}), 401
        else:
            # Update device state for non-Spotify commands
            update_device_state(command, params)
            response_text = get_dynamic_response(command, params)

        # Log the command with timestamp
        log_entry = {
            "command": command,
            "params": params,
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save log to file
        logs = load_logs()
        logs.append(log_entry)
        try:
            with open(LOG_FILE, "w") as file:
                json.dump(logs, file, indent=4)
        except Exception as e:
            logger.error(f"Error saving log: {str(e)}")
            return jsonify({"status": "warning", "message": response_text, 
                           "log_error": "Command executed but logging failed"}), 200
        
        return jsonify({"status": "success", "message": response_text})
    
    except Exception as e:
        logger.error(f"Error in command handler: {str(e)}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route("/search_songs", methods=["POST"])
def search_songs():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"status": "error", "message": "No search query provided"}), 400
            
        sp = get_spotify_client()
        if not sp:
            return jsonify({"status": "error", "message": "Spotify not authenticated"}), 401
            
        results = sp.search(q=query, type="track", limit=10)
        tracks = results["tracks"]["items"]
        
        songs = []
        for track in tracks:
            songs.append({
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "album": track["album"]["name"],
                "uri": track["uri"],
                "image": track["album"]["images"][1]["url"] if track["album"]["images"] else ""
            })
            
        return jsonify({"status": "success", "songs": songs})
    except Exception as e:
        logger.error(f"Error searching songs: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500
# Health check endpoint with device states
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "devices": DEVICE_STATE
    })

@app.route("/command/text", methods=["POST"])
def handle_text_command():
    try:
        data = request.get_json()
        command = data.get("command", "").lower()
        intent, params = predict_intent(command)
        
        if intent == "set_ac_temperature":
            temp = params.get("temperature")
            if temp:
                update_device_state(intent, params)
                response_text = f"🌡️ Temperature set to {temp}°C. Adjusting climate control..."
            else:
                return jsonify({"status": "error", "message": "Please specify a valid temperature (16-30°C)"}), 400
        else:
            update_device_state(intent, params)
            response_text = get_dynamic_response(intent, params)
            
        return jsonify({"status": "success", "message": response_text})
    except Exception as e:
        logger.error(f"Error in text command handler: {str(e)}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

# Spotify OAuth callback
@app.route("/callback")
def callback():
    try:
        code = request.args.get('code')
        token_info = sp_oauth.get_access_token(code, check_cache=False)
        logger.info("Spotify authenticated successfully")
        return "Spotify authenticated! Return to terminal."
    except Exception as e:
        logger.error(f"Spotify callback error: {str(e)}")
        return f"Authentication failed: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
