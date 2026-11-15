
import cv2
import mediapipe as mp
import time
import os
import math

# -------------------------------
# MODEL FILE LOCATION
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Model file not found!")
    print("Looking at:", MODEL_PATH)
    exit()

print("Model file found successfully!")

# -------------------------------
# INITIALIZE MEDIAPIPE
# -------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# -------------------------------
# EYE LANDMARK INDICES
# -------------------------------

# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------------
# EAR CALCULATION FUNCTION
# -------------------------------

def calculate_ear(landmarks, eye_indices, width, height):

    points = []

    for index in eye_indices:
        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Vertical distances
    A = math.dist(points[1], points[5])
    B = math.dist(points[2], points[4])

    # Horizontal distance
    C = math.dist(points[0], points[3])

    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)

    return ear, points


# -------------------------------
# BLINK DETECTION SETTINGS
# -------------------------------

EAR_THRESHOLD = 0.21

MIN_CLOSED_FRAMES = 2

blink_count = 0
closed_frames = 0
eyes_closed = False

# -------------------------------
# OPEN WEBCAM
# -------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam!")
    landmarker.close()
    exit()

last_timestamp = 0

# -------------------------------
# MAIN LOOP
# -------------------------------

while True:

    ok, frame = cap.read()

    if not ok:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape

    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = max(
        int(time.monotonic() * 1000),
        last_timestamp + 1
    )

    last_timestamp = timestamp

    results = landmarker.detect_for_video(
        mp_image, timestamp
    )

    if results.face_landmarks:

        face = results.face_landmarks[0]

        # Draw all facial landmarks
        for landmark in face:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        # Calculate EAR for both eyes
        right_ear, right_points = calculate_ear(
            face, RIGHT_EYE, width, height
        )

        left_ear, left_points = calculate_ear(
            face, LEFT_EYE, width, height
        )

        # Average EAR
        avg_ear = (right_ear + left_ear) / 2.0

        # Draw eye landmark points
        for point in right_points + left_points:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )

        # Check whether eyes are closed
        if avg_ear < EAR_THRESHOLD:

            closed_frames += 1
            eyes_closed = True

        else:

            # Count blink when eyes reopen
            if eyes_closed and closed_frames >= MIN_CLOSED_FRAMES:
                blink_count += 1

            closed_frames = 0
            eyes_closed = False

        # Display eye status
        eye_status = "CLOSED" if avg_ear < EAR_THRESHOLD else "OPEN"

        cv2.putText(
            frame,
            f"EAR: {avg_ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if eye_status == "CLOSED" else (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blink Count: {blink_count}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Display webcam
    cv2.imshow("NeuroFocus - Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Total blinks detected:", blink_count)
