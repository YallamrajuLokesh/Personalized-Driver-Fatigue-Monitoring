import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import time
import os
import math
import json

# --- JSON Logging Config ---
STATUS_FILE = "status.json"
ALERTS_FILE = "alerts.json"
PROFILES_FILE = "profiles.json" # New: for user profiles

def write_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

def append_alert(alert):
    try:
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        alerts = []
    alerts.append(alert)
    alerts = alerts[-20:]  # keep last 20
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f)

# --- User Profile Management ---
def load_profiles():
    try:
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

profiles = load_profiles()
current_user_profile = None

# --- Constants (Default values, will be overridden by profile) ---
# Drowsiness detection
EAR_THRESHOLD = 0.21 # Default until calibrated
EYE_CLOSED_FRAMES = 25

# Yawn detection
MAR_THRESHOLD = 0.75 # Default until calibrated
MOUTH_AREA_MIN = 1500
MOUTH_AREA_MAX = 25000
MOUTH_RATIO_THRESHOLD = 0.35
YAWN_DETECTION_TIME = 0.5

# Head pose detection
HEAD_TILT_THRESHOLD = 15  # degrees (deviation from neutral)
HEAD_TILT_FRAMES = 20

PITCH_UP_THRESHOLD = 20   # degrees (deviation from neutral)
PITCH_DOWN_THRESHOLD = -20 # degrees (deviation from neutral)
PITCH_UP_ALLOW = 3
PITCH_DOWN_ALLOW = 2

YAW_THRESHOLD = 20
YAW_ALLOW = 6

# Calibration state variables
calibration_frames = 0
calibration_ear_max = 0 # Changed to max for EAR
calibration_mar_max = 0 # Still max for MAR
calibration_pitch_sum = 0
calibration_roll_sum = 0
calibration_phase = 0 # 0: None, 1: EAR, 2: MAR, 3: Pitch/Roll
CALIBRATION_TOTAL_FRAMES = 60 # Number of frames for each calibration step

# --- Alarm ---
ALARM_FILE = "alarm.wav"

# Head tilt detection parameters
head_tilt_counter = 0
head_tilt_alarm_on = False

# Pitch/Yaw detection parameters
pitch_up_start = None
pitch_down_start = None
pitch_alarm_on = False
NEUTRAL_PITCH = None # Will be set during calibration

yaw_left_start = None
yaw_right_start = None
yaw_alarm_on = False

NEUTRAL_ROLL = None # Will be set during calibration

# Fixed 3D model points for better head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip (point 30)
    (0.0, -330.0, -65.0),       # Chin (point 8)
    (-225.0, 170.0, -135.0),    # Left eye left corner (point 36)
    (225.0, 170.0, -135.0),     # Right eye right corner (point 45)
    (-150.0, -150.0, -125.0),   # Left Mouth corner (point 48)
    (150.0, -150.0, -125.0)     # Right mouth corner (point 54)
])

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
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# Alternative pitch calculation using facial landmarks geometry
def calculate_pitch_from_landmarks(landmarks):
    # Get key points
    nose_tip = np.array(landmarks[30])
    chin = np.array(landmarks[8])
    nose_bridge = np.array(landmarks[27])

    # Calculate vectors
    face_vertical = chin - nose_bridge
    nose_vector = nose_tip - nose_bridge

    # Calculate angle between vectors
    dot_product = np.dot(face_vertical, nose_vector)
    magnitude_face = np.linalg.norm(face_vertical)
    magnitude_nose = np.linalg.norm(nose_vector)

    if magnitude_face > 0 and magnitude_nose > 0:
        cos_angle = dot_product / (magnitude_face * magnitude_nose)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        pitch_degrees = np.degrees(angle) - 90

        # Adjust sign based on nose position relative to face center
        if nose_tip[1] < nose_bridge[1]: # Nose tip above bridge (head up)
            pitch_degrees = abs(pitch_degrees)
        else: # Nose tip below bridge (head down)
            pitch_degrees = -abs(pitch_degrees)

        return pitch_degrees
    return 0

# Initialize pygame
pygame.mixer.init()
if os.path.exists(ALARM_FILE):
    pygame.mixer.music.load(ALARM_FILE)
else:
    print(f"Alarm file not found at {os.path.abspath(ALARM_FILE)}")

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10]) # 51, 59
    B = distance.euclidean(mouth[4], mouth[8]) # 53, 57
    C = distance.euclidean(mouth[0], mouth[6]) # 48, 54
    return (A + B) / (2.0 * C)

def calculate_mouth_area(mouth_points):
    mouth_points = np.array(mouth_points, dtype=np.int32)
    hull = cv2.convexHull(mouth_points)
    return cv2.contourArea(hull)

def play_alarm(is_yawn=False):
    try:
        if not pygame.mixer.music.get_busy(): # Only play if not already playing
            pygame.mixer.music.set_volume(0.3 if is_yawn else 1.0)
            pygame.mixer.music.play()
    except Exception as e:
        print("Audio playback error:", e)

cap = cv2.VideoCapture(0)
eye_closed_counter = 0
eye_alarm_on = False
yawn_alarm_on = False
yawn_start_time = 0
show_debug_info = True
last_mar = 0
last_mouth_area = 0
last_mouth_ratio = 0

# --- User Profile Selection Loop ---
def select_user_profile():
    global current_user_profile, EAR_THRESHOLD, MAR_THRESHOLD, NEUTRAL_PITCH, NEUTRAL_ROLL

    while True:
        print("\n--- User Profile Management ---")
        print("1. Create New User Profile")
        if profiles: # Only show if there are existing profiles
            print("2. Load Existing User Profile")
        print("Q. Quit Application")

        choice = input("Enter your choice: ").strip().lower()

        if choice == '1':
            username = input("Enter new username: ").strip()
            if username in profiles:
                print(f"Profile '{username}' already exists. Please choose a different name or load it.")
                continue
            profiles[username] = {
                "EAR_THRESHOLD": 0.21, # Default until calibrated
                "MAR_THRESHOLD": 0.75, # Default until calibrated
                "NEUTRAL_PITCH": None,
                "NEUTRAL_ROLL": None
            }
            current_user_profile = username
            save_profiles(profiles)
            print(f"Profile '{username}' created. Please proceed to calibration.")
            return True # Proceed to calibration
        elif choice == '2' and profiles:
            print("\nExisting Profiles:")
            for i, user in enumerate(profiles.keys()):
                print(f"{i+1}. {user}")
            user_choice = input("Enter number of profile to load: ").strip()
            try:
                user_index = int(user_choice) - 1
                if 0 <= user_index < len(profiles):
                    username = list(profiles.keys())[user_index]
                    current_user_profile = username
                    user_data = profiles[username]
                    EAR_THRESHOLD = user_data.get("EAR_THRESHOLD", 0.21)
                    MAR_THRESHOLD = user_data.get("MAR_THRESHOLD", 0.75)
                    NEUTRAL_PITCH = user_data.get("NEUTRAL_PITCH")
                    NEUTRAL_ROLL = user_data.get("NEUTRAL_ROLL")
                    print(f"Profile '{username}' loaded.")
                    return True # Proceed to detection
                else:
                    print("Invalid profile number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == 'q':
            print("Exiting application.")
            return False # Exit
        else:
            print("Invalid choice.")

if not select_user_profile():
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    exit()

# --- Main loop starts here after profile selection ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    display_text_y_offset = 30 # For alerts
    calibration_message = ""
    # --- Calibration Phase ---
    # Check if any crucial personalized threshold is still at its default/None value
    if current_user_profile and (profiles[current_user_profile].get("EAR_THRESHOLD", 0.21) == 0.21 or \
                                 profiles[current_user_profile].get("MAR_THRESHOLD", 0.75) == 0.75 or \
                                 profiles[current_user_profile].get("NEUTRAL_PITCH") is None or \
                                 profiles[current_user_profile].get("NEUTRAL_ROLL") is None):

        if calibration_phase == 0: # Start calibration sequence if not already set
            calibration_phase = 1 # Start with EAR calibration
            calibration_frames = 0
            calibration_ear_max = 0 # Reset for new calibration run
            calibration_mar_max = 0 # Reset for new calibration run
            calibration_pitch_sum = 0
            calibration_roll_sum = 0
            print(f"Starting calibration for user: {current_user_profile}")

        if not faces:
            calibration_message = "No face detected for calibration. Please ensure your face is visible."
            cv2.putText(frame, calibration_message, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Driver Fatigue Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue # Skip the rest of the detection logic during calibration
        else:
            # Only process calibration if a face is detected
            face = faces[0] # Assume largest face if multiple
            landmarks = predictor(gray, face)
            landmarks = [(p.x, p.y) for p in landmarks.parts()]

            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            mouth = landmarks[48:68]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            mouth_width = distance.euclidean(mouth[0], mouth[6])
            mouth_height = (distance.euclidean(mouth[2], mouth[10]) + distance.euclidean(mouth[4], mouth[8])) / 2
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0


            # Head pose for pitch and roll
            image_points = np.array([
                landmarks[30],  # Nose tip
                landmarks[8],   # Chin
                landmarks[36],  # Left eye left corner
                landmarks[45],  # Right eye right corner
                landmarks[48],  # Left Mouth corner
                landmarks[54]   # Right mouth corner
            ], dtype="double")

            size = frame.shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = rotationMatrixToEulerAngles(rotation_matrix)

            pitch_landmark = calculate_pitch_from_landmarks(landmarks)
            pitch_pnp = np.degrees(euler_angles[0])
            pitch = (pitch_landmark + pitch_pnp) / 2.0 # Combined pitch
            roll = np.degrees(euler_angles[2])

            if calibration_phase == 1: # EAR Calibration: Keep eyes open
                calibration_ear_max = max(calibration_ear_max, ear) # Track max EAR
                calibration_frames += 1
                calibration_message = f"Calibrating EAR: Keep your eyes WIDE OPEN. Max EAR: {calibration_ear_max:.2f} ({calibration_frames}/{CALIBRATION_TOTAL_FRAMES})"
                if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
                    # Set EAR_THRESHOLD as a percentage of the max observed open eye EAR
                    EAR_THRESHOLD = calibration_ear_max * 0.75 # Adjusted to 75%
                    profiles[current_user_profile]["EAR_THRESHOLD"] = EAR_THRESHOLD
                    save_profiles(profiles)
                    print(f"EAR Calibration complete. Personalized EAR_THRESHOLD: {EAR_THRESHOLD:.2f}")
                    calibration_phase = 2 # Next phase
                    calibration_frames = 0
                    calibration_mar_max = 0 # Reset for next phase
            elif calibration_phase == 2: # MAR Calibration: Yawn or open mouth wide
                calibration_mar_max = max(calibration_mar_max, mar) # Track max MAR during yawn
                calibration_frames += 1
                calibration_message = f"Calibrating MAR: Perform a few NATURAL YAWNS. Max MAR: {calibration_mar_max:.2f} ({calibration_frames}/{CALIBRATION_TOTAL_FRAMES})"
                if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
                    # Set MAR_THRESHOLD as a percentage of the max observed yawn MAR
                    MAR_THRESHOLD = calibration_mar_max * 0.85 # Good starting point, adjust if needed
                    profiles[current_user_profile]["MAR_THRESHOLD"] = MAR_THRESHOLD
                    save_profiles(profiles)
                    print(f"MAR Calibration complete. Personalized MAR_THRESHOLD: {MAR_THRESHOLD:.2f}")
                    calibration_phase = 3 # Next phase
                    calibration_frames = 0
                    calibration_pitch_sum = 0
                    calibration_roll_sum = 0
            elif calibration_phase == 3: # Pitch/Roll Calibration: Neutral head position
                calibration_pitch_sum += pitch
                calibration_roll_sum += roll
                calibration_frames += 1
                calibration_message = f"Calibrating Head Pose: Sit straight, look ahead. {calibration_frames}/{CALIBRATION_TOTAL_FRAMES}"
                if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
                    NEUTRAL_PITCH = calibration_pitch_sum / CALIBRATION_TOTAL_FRAMES
                    NEUTRAL_ROLL = calibration_roll_sum / CALIBRATION_TOTAL_FRAMES
                    profiles[current_user_profile]["NEUTRAL_PITCH"] = NEUTRAL_PITCH
                    profiles[current_user_profile]["NEUTRAL_ROLL"] = NEUTRAL_ROLL
                    save_profiles(profiles)
                    print(f"Pitch/Roll Calibration complete. Neutral Pitch: {NEUTRAL_PITCH:.2f}, Neutral Roll: {NEUTRAL_ROLL:.2f}")
                    calibration_phase = 0 # Calibration finished

        cv2.putText(frame, calibration_message, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Driver Fatigue Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue # Skip the rest of the detection logic during calibration

    # --- Detection Logic (only runs if calibration is complete or profile loaded) ---
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        mouth = landmarks[48:68]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        mouth_area = calculate_mouth_area(mouth)
        mouth_width = distance.euclidean(mouth[0], mouth[6])
        mouth_height = (distance.euclidean(mouth[2], mouth[10]) + distance.euclidean(mouth[4], mouth[8])) / 2

        # Avoid division by zero if mouth_width is too small
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0


        last_mar = mar
        last_mouth_area = mouth_area
        last_mouth_ratio = mouth_ratio

        # Eye closure detection (using personalized EAR_THRESHOLD)
        if ear < EAR_THRESHOLD:
            eye_closed_counter += 1
        else:
            eye_closed_counter = max(0, eye_closed_counter - 2)

        # Improved yawn detection (using personalized MAR_THRESHOLD)
        is_yawning = (
            mar > MAR_THRESHOLD and
            MOUTH_AREA_MIN < mouth_area < MOUTH_AREA_MAX and
            mouth_ratio > MOUTH_RATIO_THRESHOLD and
            mouth_width > 50
        )

        if is_yawning:
            if yawn_start_time == 0:
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time >= YAWN_DETECTION_TIME:
                cv2.putText(frame, "YAWN DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if not yawn_alarm_on:
                    play_alarm(is_yawn=True)
                    yawn_alarm_on = True
        else:
            yawn_start_time = 0
            yawn_alarm_on = False

        if eye_closed_counter > EYE_CLOSED_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not eye_alarm_on:
                play_alarm(is_yawn=False)
                eye_alarm_on = True
        else:
            eye_alarm_on = False

        # --- HEAD POSE ESTIMATION ---
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left Mouth corner
            landmarks[54]   # Right mouth corner
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = rotationMatrixToEulerAngles(rotation_matrix)

        pitch_landmark = calculate_pitch_from_landmarks(landmarks)
        pitch_pnp = np.degrees(euler_angles[0])

        # Combine both methods for more robust detection
        pitch = (pitch_landmark + pitch_pnp) / 2.0
        yaw = np.degrees(euler_angles[1])
        roll = np.degrees(euler_angles[2])

        # Display neutral pitch and relative pitch
        if NEUTRAL_PITCH is not None:
            relative_pitch = pitch - NEUTRAL_PITCH
            cv2.putText(frame, f"Neutral P: {NEUTRAL_PITCH:.2f}", (frame.shape[1] - 300, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Rel Pitch: {relative_pitch:.2f}", (frame.shape[1] - 300, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            relative_pitch = pitch # Use absolute pitch if not calibrated, though alerts won't fire for it without neutral_pitch
            cv2.putText(frame, "P: Not Calibrated", (frame.shape[1] - 300, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1) # Orange color for uncalibrated

        if NEUTRAL_ROLL is not None:
            relative_roll = roll - NEUTRAL_ROLL
            cv2.putText(frame, f"Neutral R: {NEUTRAL_ROLL:.2f}", (frame.shape[1] - 300, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Rel Roll: {relative_roll:.2f}", (frame.shape[1] - 300, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            relative_roll = roll # Use absolute roll if not calibrated, though alerts won't fire for it without neutral_roll
            cv2.putText(frame, "R: Not Calibrated", (frame.shape[1] - 300, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1) # Orange color for uncalibrated


        cv2.putText(frame, f"Pitch: {pitch:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Roll: {roll:.2f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        current_time = time.time()

        # --- PITCH ALERT LOGIC ---
        if NEUTRAL_PITCH is not None: # Only use relative pitch if calibrated
            if relative_pitch > PITCH_UP_THRESHOLD:
                if pitch_up_start is None:
                    pitch_up_start = current_time
                if current_time - pitch_up_start > PITCH_UP_ALLOW:
                    cv2.putText(frame, "HEAD UP ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not pitch_alarm_on:
                        play_alarm(is_yawn=False)
                        pitch_alarm_on = True
            else:
                pitch_up_start = None

            if relative_pitch < PITCH_DOWN_THRESHOLD:
                if pitch_down_start is None:
                    pitch_down_start = current_time
                if current_time - pitch_down_start > PITCH_DOWN_ALLOW:
                    cv2.putText(frame, "HEAD DOWN ALERT!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not pitch_alarm_on:
                        play_alarm(is_yawn=False)
                        pitch_alarm_on = True
            else:
                pitch_down_start = None

            if (pitch_up_start is None and pitch_down_start is None):
                pitch_alarm_on = False
        # else: # If not calibrated, pitch alerts are effectively off.

        # --- YAW ALERT LOGIC ---
        if yaw > YAW_THRESHOLD:
            if yaw_right_start is None:
                yaw_right_start = current_time
            if current_time - yaw_right_start > YAW_ALLOW:
                cv2.putText(frame, "LOOKING RIGHT ALERT!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not yaw_alarm_on:
                    play_alarm(is_yawn=False)
                    yaw_alarm_on = True
        else:
            yaw_right_start = None

        if yaw < -YAW_THRESHOLD:
            if yaw_left_start is None:
                yaw_left_start = current_time
            if current_time - yaw_left_start > YAW_ALLOW:
                cv2.putText(frame, "LOOKING LEFT ALERT!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not yaw_alarm_on:
                    play_alarm(is_yawn=False)
                    yaw_alarm_on = True
        else:
            yaw_left_start = None

        if (yaw_left_start is None and yaw_right_start is None):
            yaw_alarm_on = False

        # --- ROLL ALERT LOGIC (head tilt) ---
        if NEUTRAL_ROLL is not None: # Only use relative roll if calibrated
            if abs(relative_roll) > HEAD_TILT_THRESHOLD:
                head_tilt_counter += 1
            else:
                head_tilt_counter = max(0, head_tilt_counter - 2)

            if head_tilt_counter > HEAD_TILT_FRAMES:
                cv2.putText(frame, "HEAD TILT ALERT!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not head_tilt_alarm_on:
                    play_alarm(is_yawn=False)
                    head_tilt_alarm_on = True
            else:
                head_tilt_alarm_on = False
        # else: # If not calibrated, roll alerts are effectively off.


        # Drawing and visualization
        eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
        mouth_color = (0, 255, 0) if mar <= MAR_THRESHOLD else (0, 0, 255)

        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (int(x), int(y)), 2, eye_color, -1)
        for (x, y) in mouth:
            cv2.circle(frame, (int(x), int(y)), 2, mouth_color, -1)
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, eye_color, 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, eye_color, 1)
        cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], True, mouth_color, 1)

        # Display metrics
        cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (frame.shape[1] - 300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
        cv2.putText(frame, f"Mouth Area: {mouth_area:.0f}", (frame.shape[1] - 300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Ratio: {mouth_ratio:.2f}", (frame.shape[1] - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if show_debug_info:
            cv2.putText(frame, f"EAR Threshold: {EAR_THRESHOLD:.2f}", (10, frame.shape[0] - 100),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"MAR Threshold: {MAR_THRESHOLD:.2f}", (10, frame.shape[0] - 70),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Head Tilt Threshold: {HEAD_TILT_THRESHOLD:.2f}", (10, frame.shape[0] - 40),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Press 'm' for mouth metrics, 'c' to recalibrate", (10, frame.shape[0] - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- LOG STATUS AND ALERTS (for dashboard) ---
    # Prepare status dictionary with default values if face not detected to avoid KeyError
    status = {
        "current_user": current_user_profile,
        "ear": float(ear) if 'ear' in locals() and faces else 0.0,
        "mar": float(mar) if 'mar' in locals() and faces else 0.0,
        "mouth_area": float(mouth_area) if 'mouth_area' in locals() and faces else 0.0,
        "mouth_ratio": float(mouth_ratio) if 'mouth_ratio' in locals() and faces else 0.0,
        "pitch": float(pitch) if 'pitch' in locals() and faces else 0.0,
        "yaw": float(yaw) if 'yaw' in locals() and faces else 0.0,
        "roll": float(roll) if 'roll' in locals() and faces else 0.0,
        "eye_alarm": eye_alarm_on,
        "yawn_alarm": yawn_alarm_on,
        "pitch_alarm": pitch_alarm_on,
        "yaw_alarm": yaw_alarm_on,
        "head_tilt_alarm": head_tilt_alarm_on,
        "EAR_THRESHOLD": float(EAR_THRESHOLD),
        "MAR_THRESHOLD": float(MAR_THRESHOLD),
        "HEAD_TILT_THRESHOLD": float(HEAD_TILT_THRESHOLD),
        "NEUTRAL_PITCH": float(NEUTRAL_PITCH) if NEUTRAL_PITCH is not None else None,
        "NEUTRAL_ROLL": float(NEUTRAL_ROLL) if NEUTRAL_ROLL is not None else None
    }
    
    # Load previous status to detect transitions for alerts
    try:
        with open(STATUS_FILE, "r") as f:
            prev_status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        prev_status = {}

    write_status(status)

    # Log alerts only on transition from non-alarm to alarm state
    if eye_alarm_on and not prev_status.get("eye_alarm", False):
        append_alert({"type": "Drowsiness", "time": time.strftime("%H:%M:%S"), "value": round(status["ear"], 2)})
    if yawn_alarm_on and not prev_status.get("yawn_alarm", False):
        append_alert({"type": "Yawn", "time": time.strftime("%H:%M:%S"), "value": round(status["mar"], 2)})
    if pitch_alarm_on and not prev_status.get("pitch_alarm", False):
        alert_type = "Head Up" if (status["pitch"] - status.get("NEUTRAL_PITCH", 0)) > 0 else "Head Down" # Use relative pitch for type
        append_alert({"type": alert_type, "time": time.strftime("%H:%M:%S"), "value": round(status["pitch"], 2)})
    if yaw_alarm_on and not prev_status.get("yaw_alarm", False):
        alert_type = "Looking Right" if status["yaw"] > 0 else "Looking Left"
        append_alert({"type": alert_type, "time": time.strftime("%H:%M:%S"), "value": round(status["yaw"], 2)})
    if head_tilt_alarm_on and not prev_status.get("head_tilt_alarm", False):
        append_alert({"type": "Head Tilt", "time": time.strftime("%H:%M:%S"), "value": round(status["roll"], 2)})


    cv2.imshow("Driver Fatigue Detection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('d'):
        show_debug_info = not show_debug_info
    elif key & 0xFF == ord('m'):
        print(f"Current Mouth Metrics - MAR: {last_mar:.2f}, Area: {last_mouth_area:.0f}, Ratio: {last_mouth_ratio:.2f}")
    elif key & 0xFF == ord('c'):
        # Trigger full recalibration for the current user
        profiles[current_user_profile]["EAR_THRESHOLD"] = 0.21 # Reset to default to force recalibration
        profiles[current_user_profile]["MAR_THRESHOLD"] = 0.75 # Reset to default to force recalibration
        profiles[current_user_profile]["NEUTRAL_PITCH"] = None
        profiles[current_user_profile]["NEUTRAL_ROLL"] = None
        save_profiles(profiles)
        calibration_phase = 0 # Reset calibration state to restart from EAR
        calibration_ear_max = 0 # Reset max for new EAR calibration
        calibration_mar_max = 0 # Reset max for new MAR calibration
        print("Recalibrating profile. Please follow on-screen instructions.")


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()