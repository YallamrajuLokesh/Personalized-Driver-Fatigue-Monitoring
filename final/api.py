from flask import Flask, jsonify, request
import json
import os
from flask_cors import CORS
from pymongo import MongoClient
from bson.json_util import dumps # For serializing MongoDB ObjectId to JSON
from dotenv import load_dotenv # ADD THIS LINE

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load environment variables from .env file
load_dotenv() # ADD THIS LINE

# --- MongoDB Configuration ---
# Get the connection string from environment variables
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING") # CHANGE THIS LINE

# Ensure the connection string is loaded
if not MONGO_CONNECTION_STRING: # ADD THIS BLOCK FOR ERROR HANDLING
    print("Error: MONGO_CONNECTION_STRING not found in .env or environment variables.")
    exit("Exiting: MongoDB connection string is essential for the API.")


DB_NAME = "fatigue_detection_db"
PROFILES_COLLECTION = "profiles"
ALERTS_COLLECTION = "alerts"
STATUS_FILE = "status.json" # Flask API reads live status from this file (written by main.py)
COMMAND_FILE = "command.json" # Flask API writes commands to this file for main.py to read

try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    profiles_collection = db[PROFILES_COLLECTION]
    alerts_collection = db[ALERTS_COLLECTION]
    print("API: MongoDB connected successfully!")
except Exception as e:
    print(f"API: Error connecting to MongoDB: {e}")
    # Consider what to do if the DB connection fails for the API
    # For now, it will likely cause issues if not connected.
    exit(f"Exiting due to MongoDB connection error: {e}") # Added exit for clarity


# --- API Routes ---

@app.route('/')
def index():
    return "Fatigue Detection API (MongoDB Backend)"

@app.route('/status')
def get_status():
    """Reads the current live status from status.json (written by main.py)."""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            try:
                status_data = json.load(f)
                return jsonify(status_data)
            except json.JSONDecodeError:
                # If file is empty or corrupted, return a default empty status
                return jsonify({
                    "current_user": None, "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "eye_alarm": False, "yawn_alarm": False, "pitch_alarm": False, "yaw_alarm": False, "head_tilt_alarm": False,
                    "EAR_THRESHOLD": 0.21, "MAR_THRESHOLD": 0.75, "HEAD_TILT_THRESHOLD": 15.0,
                    "NEUTRAL_PITCH": None, "NEUTRAL_ROLL": None
                })
    # If status.json does not exist, return a default "no user" status
    return jsonify({
        "current_user": None, "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "eye_alarm": False, "yawn_alarm": False, "pitch_alarm": False, "yaw_alarm": False, "head_tilt_alarm": False,
        "EAR_THRESHOLD": 0.21, "MAR_THRESHOLD": 0.75, "HEAD_TILT_THRESHOLD": 15.0,
        "NEUTRAL_PITCH": None, "NEUTRAL_ROLL": None
    })


@app.route('/alerts')
def get_alerts():
    """
    Fetches recent alerts from MongoDB for the currently active user (if any),
    or general recent alerts if no user is active.
    """
    current_user = None
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            try:
                current_status = json.load(f)
                current_user = current_status.get("current_user")
            except json.JSONDecodeError:
                pass # current_user remains None

    if current_user:
        # Fetch alerts for the currently active user, sorted by time descending, limit to 20
        alerts_cursor = alerts_collection.find({"username": current_user}).sort("_id", -1).limit(20) # Sort by _id for newest first
    else:
        # If no user is active, fetch general recent alerts (or none)
        alerts_cursor = alerts_collection.find({}).sort("_id", -1).limit(20) # Sort by _id for newest first

    alerts_list = list(alerts_cursor)
    # dumps handles MongoDB's ObjectId serialization for jsonify
    return dumps(alerts_list)


@app.route('/profiles')
def get_profiles():
    """Fetches a list of all profile usernames from MongoDB."""
    # Only retrieve the 'username' field to minimize data transfer
    profile_names = [doc["username"] for doc in profiles_collection.find({}, {"username": 1})]
    return jsonify(profile_names)

@app.route('/profiles/create', methods=['POST'])
def create_profile():
    """Creates a new user profile in MongoDB and sends a command to main.py to select it."""
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Check if profile already exists in DB
    if profiles_collection.find_one({"username": username}):
        return jsonify({"error": f"Profile '{username}' already exists"}), 409

    # Create new profile data with default thresholds
    new_profile_data = {
        "username": username,
        "EAR_THRESHOLD": 0.21,
        "MAR_THRESHOLD": 0.75,
        "NEUTRAL_PITCH": None,
        "NEUTRAL_ROLL": None
    }
    profiles_collection.insert_one(new_profile_data) # Insert into MongoDB

    # Send command to main.py to load this newly created profile
    with open(COMMAND_FILE, 'w') as f:
        json.dump({"action": "load_profile", "username": username}, f)

    return jsonify({"message": f"Profile '{username}' created and selected for main.py"}), 201

@app.route('/profiles/select', methods=['POST'])
def select_profile():
    """Sends a command to main.py to select an existing user profile."""
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Check if profile exists in DB before attempting to select
    if not profiles_collection.find_one({"username": username}):
        return jsonify({"error": f"Profile '{username}' not found"}), 404

    # Send command to main.py to load this profile
    with open(COMMAND_FILE, 'w') as f:
        json.dump({"action": "load_profile", "username": username}, f)

    return jsonify({"message": f"Profile '{username}' selected for main.py"}), 200

@app.route('/profiles/recalibrate', methods=['POST'])
def recalibrate_profile():
    """Sends a command to main.py to initiate recalibration for the currently active user."""
    # main.py handles the actual resetting of thresholds and re-entering calibration mode.
    # We just send a simple trigger command.
    with open(COMMAND_FILE, 'w') as f:
        json.dump({"action": "recalibrate"}, f)
    return jsonify({"message": "Recalibration command sent to main.py"}), 200

if __name__ == '__main__':
    # Ensure command file exists for inter-process communication
    if not os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, 'w') as f:
            json.dump({}, f)
    # Ensure status file exists for main.py to write to
    if not os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'w') as f:
            json.dump({"current_user": None}, f)

    app.run(debug=True, port=5000)