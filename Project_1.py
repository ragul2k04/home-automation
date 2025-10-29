from flask import Flask, request, jsonify

app = Flask(__name__)

# Default route to check if Flask is running
@app.route('/')
def home():
    return "Flask is running and ready for commands!"

# Route to handle voice commands
@app.route('/command', methods=['POST'])
def command():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Extract command from request
    command = data.get('command')
    if command:
        response = process_command(command)
        return jsonify({"response": response}), 200
    else:
        return jsonify({"error": "No command found"}), 400

def process_command(command):
    # Handle different commands dynamically
    command = command.lower()
    if "turn on" in command and "light" in command:
        return "Light turned ON successfully! 💡"
    elif "turn off" in command and "light" in command:
        return "Light turned OFF successfully! 🔥"
    elif "play music" in command:
        return "Playing music now! 🎵"
    elif "stop music" in command:
        return "Music stopped! 🎧"
    else:
        return "Sorry, I didn't understand that command. ❗️"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
