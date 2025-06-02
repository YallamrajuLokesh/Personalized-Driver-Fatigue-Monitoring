import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import time
import os
import math
import json
from pymongo import MongoClient
from bson.objectid import ObjectId # Needed if you want to use MongoDB's default _id

# --- MongoDB Configuration ---
# IMPORTANT: Replace with your MongoDB connection string!
# For local MongoDB (Docker/direct install):
MONGO_CONNECTION_STRING = "mongodb+srv://root:12345@cluster0.76twovi.mongodb.net/"
# For MongoDB Atlas (cloud), it looks like:
# MONGO_CONNECTION_STRING = "mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "fatigue_detection_db"
PROFILES_COLLECTION = "profiles"
ALERTS_COLLECTION = "alerts"

try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    profiles_collection = db[PROFILES_COLLECTION]
    alerts_collection = db[ALERTS_COLLECTION]
    print("MongoDB connected successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Exit or handle error gracefully if DB connection is critical
    exit()

# --- Inter-process Communication File (still used for commands from API to main.py) ---
COMMAND_FILE = "command.json"
STATUS_FILE = "status.json" # main.py will write live status to this file for api.py to read

# --- User Profile Management ---
def load_profile_from_db(username):
    return profiles_collection.find_one({"username": username})

def save_profile_to_db(profile_data):
    # If the profile already has a MongoDB _id, update it; otherwise, insert
    if "_id" in profile_data:
        # Use update_one with $set to update specific fields, or replace_one to replace whole doc
        profiles_collection.replace_one({"_id": profile_data["_id"]}, profile_data)
    else:
        profiles_collection.insert_one(profile_data)

def append_alert_to_db(alert):
    alerts_collection.insert_one(alert)

# --- Global Variables for Current Profile ---
current_user_profile_data = None # Stores the full profile document from DB
current_user_username = None

# --- Constants (Default values, will be overridden by personalized profile data) ---
# These are fallback values if a profile is not yet calibrated.
EAR_THRESHOLD = 0.21
EYE_CLOSED_FRAMES = 25
MAR_THRESHOLD = 0.75
MOUTH_AREA_MIN = 1500 # Adjust if needed for your camera/distance
MOUTH_AREA_MAX = 25000 # Adjust if needed for your camera/distance
MOUTH_RATIO_THRESHOLD = 0.35 # Ratio of mouth height to width for yawning
YAWN_DETECTION_TIME = 0.5 # Duration (seconds) for a yawn to be considered
HEAD_TILT_THRESHOLD = 15 # Degrees from neutral for head tilt
HEAD_TILT_FRAMES = 20 # Number of frames for head tilt to be considered an alert
PITCH_UP_THRESHOLD = 20 # Degrees up from neutral pitch
PITCH_DOWN_THRESHOLD = -20 # Degrees down from neutral pitch
PITCH_UP_ALLOW = 3 # Seconds allowed for head up
PITCH_DOWN_ALLOW = 2 # Seconds allowed for head down
YAW_THRESHOLD = 20 # Degrees left/right from neutral yaw
YAW_ALLOW = 6 # Seconds allowed for head turned left/right

# Calibration state variables
calibration_frames = 0
calibration_ear_max = 0
calibration_mar_max = 0
calibration_pitch_sum = 0
calibration_roll_sum = 0
calibration_phase = 0 # 0: None, 1: EAR, 2: MAR, 3: Pitch/Roll
CALIBRATION_TOTAL_FRAMES = 60 # Number of frames for each calibration step

# --- Alarm ---
ALARM_FILE = "alarm.wav"

head_tilt_counter = 0
head_tilt_alarm_on = False
pitch_up_start = None
pitch_down_start = None
pitch_alarm_on = False
NEUTRAL_PITCH = None # Will be set by calibration
yaw_left_start = None
yaw_right_start = None
yaw_alarm_on = False
NEUTRAL_ROLL = None # Will be set by calibration

# 3D model points (standard face model)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype="double")

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else: # Gimbal lock
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z]) # Returns in radians

def calculate_pitch_from_landmarks(landmarks):
    # Using nose tip, chin, and bridge to calculate a simplified pitch
    nose_tip = np.array(landmarks[30]) # Nose Tip
    chin = np.array(landmarks[8])      # Chin
    nose_bridge = np.array(landmarks[27]) # Nose bridge (between eyes)

    # Vector from nose bridge to chin
    face_vertical = chin - nose_bridge
    # Vector from nose bridge to nose tip
    nose_vector = nose_tip - nose_bridge

    dot_product = np.dot(face_vertical, nose_vector)
    magnitude_face = np.linalg.norm(face_vertical)
    magnitude_nose = np.linalg.norm(nose_vector)

    if magnitude_face > 0 and magnitude_nose > 0:
        cos_angle = dot_product / (magnitude_face * magnitude_nose)
        cos_angle = np.clip(cos_angle, -1.0, 1.0) # Clamp to avoid floating point errors
        angle = np.arccos(cos_angle) # Angle in radians between the two vectors

        # Convert to degrees and adjust to represent typical pitch (0 for straight, positive for up, negative for down)
        pitch_degrees = np.degrees(angle) - 90
        # Refine sign based on nose tip's y-coordinate relative to nose bridge
        # If nose tip's y is less than nose bridge's y, it means head is tilted up (in image coords)
        if nose_tip[1] < nose_bridge[1]: # Nose tip is above bridge (smaller Y), means head tilted up
            pitch_degrees = abs(pitch_degrees)
        else: # Nose tip is below bridge (larger Y), means head tilted down
            pitch_degrees = -abs(pitch_degrees)
        return pitch_degrees
    return 0


# Initialize Pygame mixer for alarm
pygame.mixer.init()
if os.path.exists(ALARM_FILE):
    pygame.mixer.music.load(ALARM_FILE)
else:
    print(f"Warning: Alarm file '{ALARM_FILE}' not found at {os.path.abspath(ALARM_FILE)}")

# Initialize Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading shape predictor: {e}")
    print("Make sure 'shape_predictor_68_face_landmarks.dat' is in the same directory as main.py.")
    exit()

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10]) # 51-59 vertical
    B = distance.euclidean(mouth[4], mouth[8]) # 53-57 vertical
    C = distance.euclidean(mouth[0], mouth[6]) # 49-55 horizontal
    return (A + B) / (2.0 * C)

# Function to calculate mouth area using convex hull
def calculate_mouth_area(mouth_points):
    mouth_points_np = np.array(mouth_points, dtype=np.int32)
    # Ensure points are structured as (N, 1, 2) for contourArea
    hull = cv2.convexHull(mouth_points_np.reshape(-1, 1, 2))
    return cv2.contourArea(hull)

# Function to play alarm sound
def play_alarm(is_yawn=False):
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.set_volume(0.3 if is_yawn else 1.0) # Yawn alarm softer
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing alarm: {e}")

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Counters and flags for detection logic
eye_closed_counter = 0
eye_alarm_on = False
yawn_alarm_on = False
yawn_start_time = 0 # To track duration of yawn

# Debug info flag
show_debug_info = True

# Variables to store last calculated metrics for dashboard update
last_ear = 0.0
last_mar = 0.0
last_mouth_area = 0.0
last_mouth_ratio = 0.0
last_pitch = 0.0
last_yaw = 0.0
last_roll = 0.0

# --- Function to load a user profile by name from DB ---
def load_user_profile_by_name(username):
    global current_user_profile_data, current_user_username, EAR_THRESHOLD, MAR_THRESHOLD, NEUTRAL_PITCH, NEUTRAL_ROLL, calibration_phase, calibration_frames, calibration_ear_max, calibration_mar_max, calibration_pitch_sum, calibration_roll_sum
    profile = load_profile_from_db(username)
    if profile:
        current_user_profile_data = profile
        current_user_username = username
        EAR_THRESHOLD = profile.get("EAR_THRESHOLD", 0.21) # Use default if not calibrated
        MAR_THRESHOLD = profile.get("MAR_THRESHOLD", 0.75) # Use default if not calibrated
        NEUTRAL_PITCH = profile.get("NEUTRAL_PITCH")
        NEUTRAL_ROLL = profile.get("NEUTRAL_ROLL")
        print(f"Profile '{username}' loaded.")

        # Check if calibration data is missing for the loaded profile
        if EAR_THRESHOLD == 0.21 or MAR_THRESHOLD == 0.75 or NEUTRAL_PITCH is None or NEUTRAL_ROLL is None:
            calibration_phase = 1 # Start calibration from EAR if incomplete
            calibration_frames = 0
            calibration_ear_max = 0
            calibration_mar_max = 0
            calibration_pitch_sum = 0
            calibration_roll_sum = 0
            print(f"Profile '{username}' requires calibration. Starting calibration sequence.")
        else:
            calibration_phase = 0 # No calibration needed
            print(f"Profile '{username}' is fully calibrated. Starting detection.")
        return True
    else:
        print(f"Profile '{username}' not found in database.")
        return False

# --- Command File Management (for communication with Flask API) ---
def get_command():
    try:
        if os.path.exists(COMMAND_FILE):
            with open(COMMAND_FILE, "r") as f:
                command = json.load(f)
            # Clear command after reading to avoid repeated action
            with open(COMMAND_FILE, "w") as fw:
                json.dump({}, fw) # Overwrite with empty JSON
            return command
        return {}
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle cases where file might be empty or corrupted
        with open(COMMAND_FILE, "w") as fw:
            json.dump({}, fw) # Reset file
        return {}

def write_status(status_data):
    """Writes the current status to status.json for the API to read."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status_data, f)
    except Exception as e:
        print(f"Error writing status to {STATUS_FILE}: {e}")

# Initial state: No user loaded, so calibration phase is 0
calibration_phase = 0

# --- Main loop ---
while True:
    # 1. Check for commands from the dashboard periodically
    command = get_command()
    if command:
        if command.get("action") == "load_profile":
            username_to_load = command.get("username")
            if username_to_load:
                load_user_profile_by_name(username_to_load)
        elif command.get("action") == "recalibrate":
            if current_user_profile_data:
                # Reset current user's profile data (in memory) to defaults to force recalibration
                # These will be updated and saved to DB during calibration
                current_user_profile_data["EAR_THRESHOLD"] = 0.21
                current_user_profile_data["MAR_THRESHOLD"] = 0.75
                current_user_profile_data["NEUTRAL_PITCH"] = None
                current_user_profile_data["NEUTRAL_ROLL"] = None
                # No need to save to DB immediately here, as calibration will update it.
                # If we save here, it would set thresholds to default in DB before new calibration
                calibration_phase = 1 # Reset calibration state to restart from EAR
                calibration_frames = 0
                calibration_ear_max = 0
                calibration_mar_max = 0
                calibration_pitch_sum = 0
                calibration_roll_sum = 0
                print(f"Recalibrating profile for {current_user_username}. Please follow on-screen instructions.")
            else:
                print("No user active to recalibrate.")

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    calibration_message = ""
    # Default values for status update if no face or no user
    current_ear = 0.0
    current_mar = 0.0
    current_mouth_area = 0.0
    current_mouth_ratio = 0.0
    current_pitch = 0.0
    current_yaw = 0.0
    current_roll = 0.0
    current_eye_alarm = False
    current_yawn_alarm = False
    current_pitch_alarm = False
    current_yaw_alarm = False
    current_head_tilt_alarm = False

    if faces:
        face = faces[0] # Assume only one face for simplicity
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        mouth = landmarks[48:68]

        current_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        current_mar = mouth_aspect_ratio(mouth)
        current_mouth_area = calculate_mouth_area(mouth)
        mouth_width = distance.euclidean(mouth[0], mouth[6])
        mouth_height = (distance.euclidean(mouth[2], mouth[10]) + distance.euclidean(mouth[4], mouth[8])) / 2
        current_mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        # Pose Estimation
        image_points = np.array([
            landmarks[30], # Nose tip
            landmarks[8],  # Chin
            landmarks[36], # Left eye left corner
            landmarks[45], # Right eye right corner
            landmarks[48], # Left mouth corner
            landmarks[54]  # Right mouth corner
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = rotationMatrixToEulerAngles(rotation_matrix)

        # Combine landmark-based and PnP-based pitch for robustness
        pitch_landmark = calculate_pitch_from_landmarks(landmarks)
        pitch_pnp = np.degrees(euler_angles[0]) # Pitch from PnP
        current_pitch = (pitch_landmark + pitch_pnp) / 2.0 # Average them

        current_yaw = np.degrees(euler_angles[1]) # Yaw from PnP
        current_roll = np.degrees(euler_angles[2]) # Roll from PnP

        # Draw landmarks and bounding box
        for (x_l, y_l) in landmarks:
            cv2.circle(frame, (x_l, y_l), 1, (0, 255, 0), -1)
        # Corrected: dlib.rectangle object's methods give you the coordinates
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height()) # FIX: Removed .create()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    # --- Calibration Phase ---
    if current_user_username and calibration_phase > 0:
        if not faces:
            calibration_message = "No face detected for calibration. Please ensure your face is visible."
        else:
            # Using current_ear, current_mar, current_pitch, current_roll from above
            if calibration_phase == 1: # EAR Calibration
                calibration_ear_max = max(calibration_ear_max, current_ear)
                calibration_frames += 1
                calibration_message = f"Calibrating EAR: Keep eyes WIDE OPEN. Max EAR: {calibration_ear_max:.2f} ({calibration_frames}/{CALIBRATION_TOTAL_FRAMES})"
                if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
                    # Set threshold slightly below max open to detect actual blinks/drowsiness
                    EAR_THRESHOLD = calibration_ear_max * 0.75
                    current_user_profile_data["EAR_THRESHOLD"] = EAR_THRESHOLD
                    save_profile_to_db(current_user_profile_data) # Save to DB
                    print(f"EAR Calibration complete. Personalized EAR_THRESHOLD: {EAR_THRESHOLD:.2f}")
                    calibration_phase = 2
                    calibration_frames = 0
                    calibration_mar_max = 0
            elif calibration_phase == 2: # MAR Calibration
                calibration_mar_max = max(calibration_mar_max, current_mar)
                calibration_frames += 1
                calibration_message = f"Calibrating MAR: Perform a few NATURAL YAWNS. Max MAR: {calibration_mar_max:.2f} ({calibration_frames}/{CALIBRATION_TOTAL_FRAMES})"
                if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
                    # Set threshold slightly below max yawn to detect actual yawns
                    MAR_THRESHOLD = calibration_mar_max * 0.85
                    current_user_profile_data["MAR_THRESHOLD"] = MAR_THRESHOLD
                    save_profile_to_db(current_user_profile_data) # Save to DB
                    print(f"MAR Calibration complete. Personalized MAR_THRESHOLD: {MAR_THRESHOLD:.2f}")
                    calibration_phase = 3
                    calibration_frames = 0
                    calibration_pitch_sum = 0
                    calibration_roll_sum = 0
            elif calibration_phase == 3: # Pitch/Roll Calibration
                calibration_pitch_sum += current_pitch
                calibration_roll_sum += current_roll
                calibration_frames += 1
                calibration_message = f"Calibrating Head Pose: Sit straight, look ahead. {calibration_frames}/{CALIBRATION_TOTAL_FRAMES}"
                if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
                    NEUTRAL_PITCH = calibration_pitch_sum / CALIBRATION_TOTAL_FRAMES
                    NEUTRAL_ROLL = calibration_roll_sum / CALIBRATION_TOTAL_FRAMES
                    current_user_profile_data["NEUTRAL_PITCH"] = NEUTRAL_PITCH
                    current_user_profile_data["NEUTRAL_ROLL"] = NEUTRAL_ROLL
                    save_profile_to_db(current_user_profile_data) # Save to DB
                    print(f"Pitch/Roll Calibration complete. Neutral Pitch: {NEUTRAL_PITCH:.2f}, Neutral Roll: {NEUTRAL_ROLL:.2f}")
                    calibration_phase = 0 # Calibration finished!

        cv2.putText(frame, calibration_message, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Update status (without alarms) during calibration for dashboard
        status_to_write = {
            "current_user": current_user_username,
            "ear": float(current_ear), "mar": float(current_mar),
            "mouth_area": float(current_mouth_area), "mouth_ratio": float(current_mouth_ratio),
            "pitch": float(current_pitch), "yaw": float(current_yaw), "roll": float(current_roll),
            "eye_alarm": False, "yawn_alarm": False, "pitch_alarm": False, "yaw_alarm": False, "head_tilt_alarm": False,
            "EAR_THRESHOLD": float(EAR_THRESHOLD), # Shows the default or current ear_threshold
            "MAR_THRESHOLD": float(MAR_THRESHOLD), # Shows the default or current mar_threshold
            "HEAD_TILT_THRESHOLD": float(HEAD_TILT_THRESHOLD),
            "NEUTRAL_PITCH": float(NEUTRAL_PITCH) if NEUTRAL_PITCH is not None else None,
            "NEUTRAL_ROLL": float(NEUTRAL_ROLL) if NEUTRAL_ROLL is not None else None
        }
        write_status(status_to_write)
        cv2.imshow("Driver Fatigue Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue # Skip normal detection if in calibration

    # --- Detection Logic (only runs if calibration is complete or profile loaded) ---
    if not current_user_username or calibration_phase != 0:
        cv2.putText(frame, "Please select a user profile to start detection.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Always write status for dashboard to show no user active
        status_to_write = {
            "current_user": None, "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "eye_alarm": False, "yawn_alarm": False, "pitch_alarm": False, "yaw_alarm": False, "head_tilt_alarm": False,
            "EAR_THRESHOLD": 0.21, "MAR_THRESHOLD": 0.75, "HEAD_TILT_THRESHOLD": HEAD_TILT_THRESHOLD,
            "NEUTRAL_PITCH": None, "NEUTRAL_ROLL": None
        }
        write_status(status_to_write)
        cv2.imshow("Driver Fatigue Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # If a user is active and calibrated, apply their thresholds for detection
    # These global variables (EAR_THRESHOLD, MAR_THRESHOLD etc.) should already be updated
    # by load_user_profile_by_name or calibration_phase logic
    # Ensure they reflect the current profile's data
    EAR_THRESHOLD_ACTIVE = current_user_profile_data.get("EAR_THRESHOLD", 0.21)
    MAR_THRESHOLD_ACTIVE = current_user_profile_data.get("MAR_THRESHOLD", 0.75)
    NEUTRAL_PITCH_ACTIVE = current_user_profile_data.get("NEUTRAL_PITCH")
    NEUTRAL_ROLL_ACTIVE = current_user_profile_data.get("NEUTRAL_ROLL")


    if faces: # Re-check if face is still detected after initial check
        # Drowsiness detection
        if current_ear < EAR_THRESHOLD_ACTIVE:
            eye_closed_counter += 1
        else:
            eye_closed_counter = max(0, eye_closed_counter - 2) # Reduce counter faster

        current_eye_alarm = False
        if eye_closed_counter > EYE_CLOSED_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not eye_alarm_on:
                play_alarm(is_yawn=False)
                eye_alarm_on = True
                append_alert_to_db({"type": "Drowsiness", "time": time.strftime("%H:%M:%S"), "value": round(current_ear, 2), "username": current_user_username})
            current_eye_alarm = True
        else:
            eye_alarm_on = False

        # Yawn detection
        is_yawning = (
            current_mar > MAR_THRESHOLD_ACTIVE and
            MOUTH_AREA_MIN < current_mouth_area < MOUTH_AREA_MAX and
            current_mouth_ratio > MOUTH_RATIO_THRESHOLD and
            mouth_width > 50 # Ensure mouth is open wide enough
        )

        current_yawn_alarm = False
        if is_yawning:
            if yawn_start_time == 0:
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time >= YAWN_DETECTION_TIME:
                cv2.putText(frame, "YAWN DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if not yawn_alarm_on:
                    play_alarm(is_yawn=True)
                    yawn_alarm_on = True
                    append_alert_to_db({"type": "Yawn", "time": time.strftime("%H:%M:%S"), "value": round(current_mar, 2), "username": current_user_username})
                current_yawn_alarm = True
        else:
            yawn_start_time = 0
            yawn_alarm_on = False

        # Head Pose Alerts
        current_time = time.time()
        current_pitch_alarm = False
        current_yaw_alarm = False
        current_head_tilt_alarm = False

        # Pitch (Head Up/Down)
        if NEUTRAL_PITCH_ACTIVE is not None:
            relative_pitch = current_pitch - NEUTRAL_PITCH_ACTIVE
            if relative_pitch > PITCH_UP_THRESHOLD:
                if pitch_up_start is None: pitch_up_start = current_time
                if current_time - pitch_up_start > PITCH_UP_ALLOW:
                    cv2.putText(frame, "HEAD UP ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not pitch_alarm_on:
                        play_alarm(is_yawn=False)
                        pitch_alarm_on = True
                        append_alert_to_db({"type": "Head Up", "time": time.strftime("%H:%M:%S"), "value": round(current_pitch, 2), "username": current_user_username})
                    current_pitch_alarm = True
            else: pitch_up_start = None

            if relative_pitch < PITCH_DOWN_THRESHOLD:
                if pitch_down_start is None: pitch_down_start = current_time
                if current_time - pitch_down_start > PITCH_DOWN_ALLOW:
                    cv2.putText(frame, "HEAD DOWN ALERT!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not pitch_alarm_on:
                        play_alarm(is_yawn=False)
                        pitch_alarm_on = True
                        append_alert_to_db({"type": "Head Down", "time": time.strftime("%H:%M:%S"), "value": round(current_pitch, 2), "username": current_user_username})
                    current_pitch_alarm = True
            else: pitch_down_start = None

            if (pitch_up_start is None and pitch_down_start is None): pitch_alarm_on = False
        else: # If neutral pitch not calibrated
            cv2.putText(frame, "Pitch: Not Calibrated", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)


        # Yaw (Looking Left/Right)
        if current_yaw > YAW_THRESHOLD: # Looking Right
            if yaw_right_start is None: yaw_right_start = current_time
            if current_time - yaw_right_start > YAW_ALLOW:
                cv2.putText(frame, "LOOKING RIGHT ALERT!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not yaw_alarm_on:
                    play_alarm(is_yawn=False)
                    yaw_alarm_on = True
                    append_alert_to_db({"type": "Looking Right", "time": time.strftime("%H:%M:%S"), "value": round(current_yaw, 2), "username": current_user_username})
                current_yaw_alarm = True
        else: yaw_right_start = None

        if current_yaw < -YAW_THRESHOLD: # Looking Left
            if yaw_left_start is None: yaw_left_start = current_time
            if current_time - yaw_left_start > YAW_ALLOW:
                cv2.putText(frame, "LOOKING LEFT ALERT!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not yaw_alarm_on:
                    play_alarm(is_yawn=False)
                    yaw_alarm_on = True
                    append_alert_to_db({"type": "Looking Left", "time": time.strftime("%H:%M:%S"), "value": round(current_yaw, 2), "username": current_user_username})
                current_yaw_alarm = True
        else: yaw_left_start = None
        if (yaw_left_start is None and yaw_right_start is None): yaw_alarm_on = False

        # Roll (Head Tilt)
        if NEUTRAL_ROLL_ACTIVE is not None:
            relative_roll = current_roll - NEUTRAL_ROLL_ACTIVE
            if abs(relative_roll) > HEAD_TILT_THRESHOLD:
                head_tilt_counter += 1
            else:
                head_tilt_counter = max(0, head_tilt_counter - 2) # Reduce counter faster
            if head_tilt_counter > HEAD_TILT_FRAMES:
                cv2.putText(frame, "HEAD TILT ALERT!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not head_tilt_alarm_on:
                    play_alarm(is_yawn=False)
                    head_tilt_alarm_on = True
                    append_alert_to_db({"type": "Head Tilt", "time": time.strftime("%H:%M:%S"), "value": round(current_roll, 2), "username": current_user_username})
                current_head_tilt_alarm = True
            else: head_tilt_alarm_on = False
        else: # If neutral roll not calibrated
            cv2.putText(frame, "Roll: Not Calibrated", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)


        # Visualize eye and mouth
        eye_color = (0, 255, 0) if current_ear >= EAR_THRESHOLD_ACTIVE else (0, 0, 255)
        mouth_color = (0, 255, 0) if not is_yawning else (0, 0, 255) # Red if yawning

        for (x_l, y_l) in left_eye + right_eye: cv2.circle(frame, (int(x_l), int(y_l)), 2, eye_color, -1)
        for (x_l, y_l) in mouth: cv2.circle(frame, (int(x_l), int(y_l)), 2, mouth_color, -1)
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, eye_color, 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, eye_color, 1)
        cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], True, mouth_color, 1)

        # Display metrics on frame
        cv2.putText(frame, f"EAR: {current_ear:.2f}", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        cv2.putText(frame, f"MAR: {current_mar:.2f}", (frame.shape[1] - 300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
        cv2.putText(frame, f"Mouth Area: {current_mouth_area:.0f}", (frame.shape[1] - 300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Ratio: {current_mouth_ratio:.2f}", (frame.shape[1] - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {current_pitch:.2f} deg", (frame.shape[1] - 300, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {current_yaw:.2f} deg", (frame.shape[1] - 300, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Roll: {current_roll:.2f} deg", (frame.shape[1] - 300, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if show_debug_info:
            cv2.putText(frame, f"EAR Thresh: {EAR_THRESHOLD_ACTIVE:.2f}", (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"MAR Thresh: {MAR_THRESHOLD_ACTIVE:.2f}", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Head Tilt Thresh: {HEAD_TILT_THRESHOLD:.2f}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if NEUTRAL_PITCH_ACTIVE is not None:
                cv2.putText(frame, f"Neutral P: {NEUTRAL_PITCH_ACTIVE:.2f}", (frame.shape[1] - 300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if NEUTRAL_ROLL_ACTIVE is not None:
                cv2.putText(frame, f"Neutral R: {NEUTRAL_ROLL_ACTIVE:.2f}", (frame.shape[1] - 300, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(frame, f"Press 'd' for debug, 'c' to recalibrate", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


    # --- Live Status for Dashboard (Written to STATUS_FILE) ---
    status_to_write = {
        "current_user": current_user_username,
        "ear": float(current_ear),
        "mar": float(current_mar),
        "mouth_area": float(current_mouth_area),
        "mouth_ratio": float(current_mouth_ratio),
        "pitch": float(current_pitch),
        "yaw": float(current_yaw),
        "roll": float(current_roll),
        "eye_alarm": current_eye_alarm,
        "yawn_alarm": current_yawn_alarm,
        "pitch_alarm": current_pitch_alarm,
        "yaw_alarm": current_yaw_alarm,
        "head_tilt_alarm": current_head_tilt_alarm,
        "EAR_THRESHOLD": float(EAR_THRESHOLD_ACTIVE),
        "MAR_THRESHOLD": float(MAR_THRESHOLD_ACTIVE),
        "HEAD_TILT_THRESHOLD": float(HEAD_TILT_THRESHOLD),
        "NEUTRAL_PITCH": float(NEUTRAL_PITCH_ACTIVE) if NEUTRAL_PITCH_ACTIVE is not None else None,
        "NEUTRAL_ROLL": float(NEUTRAL_ROLL_ACTIVE) if NEUTRAL_ROLL_ACTIVE is not None else None
    }
    write_status(status_to_write)


    cv2.imshow("Driver Fatigue Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_debug_info = not show_debug_info
    elif key == ord('c'):
        if current_user_profile_data:
            # Trigger recalibration via command to self
            # This is effectively simulating a command from the API
            with open(COMMAND_FILE, 'w') as f:
                json.dump({"action": "recalibrate"}, f)
            print("Recalibration command sent (via self).")
        else:
            print("No user active to recalibrate. Please select or create a user first.")


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
if 'client' in locals(): # Ensure client was initialized before trying to close
    client.close() # Close MongoDB connection
    print("MongoDB connection closed.")