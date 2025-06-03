from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from bson.json_util import dumps
from dotenv import load_dotenv
import os
import json
import bcrypt

app = Flask(__name__)
CORS(app)

load_dotenv()

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
if not MONGO_CONNECTION_STRING:
    print("Error: MONGO_CONNECTION_STRING not found.")
    exit()

DB_NAME = "fatigue_detection_db"
ACCOUNTS_COLLECTION = "accounts"
PROFILES_COLLECTION = "profiles"
ALERTS_COLLECTION = "alerts"

# Update path resolution
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, 'data')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Update file paths to use os.path.join
STATUS_FILE = os.path.join(DATA_DIR, 'status.json')
COMMAND_FILE = os.path.join(DATA_DIR, 'command.json')

try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    accounts_collection = db[ACCOUNTS_COLLECTION]
    profiles_collection = db[PROFILES_COLLECTION]
    alerts_collection = db[ALERTS_COLLECTION]
    print("API: MongoDB connected.")
except Exception as e:
    exit(f"MongoDB error: {e}")

@app.route('/')
def index():
    return "Fatigue Detection API (MongoDB + Account Support)"

# ------------------ ACCOUNT ROUTES ------------------

@app.route('/account/register', methods=['POST'])
def register_account():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        print(f"Attempting to register user: {username}")  # Debug log
        
        if not username or not password:
            print("Missing username or password")  # Debug log
            return jsonify({'error': 'Username and password required'}), 400

        if accounts_collection.find_one({'username': username}):
            print(f"Username {username} already exists")  # Debug log
            return jsonify({'error': 'Username already exists'}), 409

        # Ensure the password is stored as a binary (BSON Binary) for MongoDB
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        result = accounts_collection.insert_one({
            'username': username, 
            'password': hashed_pw.decode('utf-8')
        })
        
        print(f"Successfully registered user: {username}")  # Debug log
        return jsonify({'message': 'Account registered successfully'}), 201
    except Exception as e:
        print(f"Registration error: {str(e)}")  # Debug log
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/account/login', methods=['POST'])
def login_account():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        print(f"Login attempt for user: {username}")  # Debug log

        account = accounts_collection.find_one({'username': username})
        if not account:
            print(f"User not found: {username}")  # Debug log
            return jsonify({'error': 'Invalid credentials'}), 401

        # Convert password back to bytes for bcrypt check
        if not bcrypt.checkpw(password.encode('utf-8'), account['password'].encode('utf-8')):
            print(f"Invalid password for user: {username}")  # Debug log
            return jsonify({'error': 'Invalid credentials'}), 401

        print(f"Successful login for user: {username}")  # Debug log
        return jsonify({'message': 'Login successful'}), 200
    except Exception as e:
        print(f"Login error: {str(e)}")  # Debug log
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

# ------------------ PROFILE ROUTES ------------------

@app.route('/profiles', methods=['GET'])
def get_profiles():
    """Return profiles belonging to a given account username"""
    username = request.args.get('account')
    if not username:
        return jsonify({'error': 'Missing account parameter'}), 400

    # Fix: Use 'account_username' as the query key (not 'username')
    profiles = profiles_collection.find({'account_username': username}, {"_id": 0, "profile_name": 1})
    profile_names = [p["profile_name"] for p in profiles]
    return jsonify(profile_names)

@app.route('/profiles/create', methods=['POST'])
def create_profile():
    data = request.json
    account_username = data.get('account_username')
    profile_name = data.get('profile_name')

    if not account_username or not profile_name:
        return jsonify({'error': 'account_username and profile_name are required'}), 400

    if profiles_collection.find_one({"account_username": account_username, "profile_name": profile_name}):
        return jsonify({'error': 'Profile already exists for this account'}), 409

    new_profile = {
        "account_username": account_username,
        "profile_name": profile_name,
        "EAR_THRESHOLD": 0.21,
        "MAR_THRESHOLD": 0.75,
        "NEUTRAL_PITCH": None,
        "NEUTRAL_ROLL": None
    }

    profiles_collection.insert_one(new_profile)

    with open(COMMAND_FILE, 'w') as f:
        json.dump({"action": "load_profile", "username": profile_name}, f)

    return jsonify({"message": f"Profile '{profile_name}' created and selected"}), 201

@app.route('/profiles/select', methods=['POST'])
def select_profile():
    data = request.json
    profile_name = data.get('profile_name')
    account_username = data.get('account_username')

    profile = profiles_collection.find_one({'account_username': account_username, 'profile_name': profile_name})
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404

    with open(COMMAND_FILE, 'w') as f:
        json.dump({"action": "load_profile", "username": profile_name}, f)

    return jsonify({'message': f"Profile '{profile_name}' selected"}), 200

@app.route('/profiles/recalibrate', methods=['POST'])
def recalibrate_profile():
    with open(COMMAND_FILE, 'w') as f:
        json.dump({"action": "recalibrate"}, f)
    return jsonify({"message": "Recalibration command sent to main.py"}), 200

@app.route('/profiles/update_calibration', methods=['POST'])
def update_calibration():
    data = request.json
    account_username = data.get('account_username')
    profile_name = data.get('profile_name')
    calibration = data.get('calibration')

    if not all([account_username, profile_name, calibration]):
        return jsonify({'error': 'Missing required fields'}), 400

    result = profiles_collection.update_one(
        {'account_username': account_username, 'profile_name': profile_name},
        {'$set': {
            'EAR_THRESHOLD': calibration['EAR_THRESHOLD'],
            'MAR_THRESHOLD': calibration['MAR_THRESHOLD'],
            'NEUTRAL_PITCH': calibration['NEUTRAL_PITCH'],
            'NEUTRAL_ROLL': calibration['NEUTRAL_ROLL']
        }}
    )

    if result.matched_count == 0:
        return jsonify({'error': 'Profile not found'}), 404

    return jsonify({'message': 'Calibration updated successfully'}), 200

# ------------------ STATUS AND ALERTS ------------------

@app.route('/status')
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            try:
                return jsonify(json.load(f))
            except json.JSONDecodeError:
                pass
    return jsonify({
        "current_user": None, "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "eye_alarm": False, "yawn_alarm": False, "pitch_alarm": False, "yaw_alarm": False, "head_tilt_alarm": False,
        "EAR_THRESHOLD": 0.21, "MAR_THRESHOLD": 0.75, "HEAD_TILT_THRESHOLD": 15.0,
        "NEUTRAL_PITCH": None, "NEUTRAL_ROLL": None
    })

@app.route('/alerts', methods=['GET'])
def get_alerts():
    try:
        account = request.args.get('account')
        
        alerts_file = os.path.join(DATA_DIR, 'alerts.json')
        with open(alerts_file, 'r') as f:
            alerts = json.load(f)
        
        if account:
            alerts = [alert for alert in alerts if alert.get('account') == account]
            
        return jsonify(alerts)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ MAIN ------------------

if __name__ == '__main__':
    if not os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'w') as f:
            json.dump({"current_user": None}, f)

    app.run(debug=True, port=5000)