import cv2
import mediapipe as mp
import time
import os

# Model file location
# Get the folder where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Locate the model file inside that folder
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Model file not found!")
    print("Looking at:", MODEL_PATH)
    exit()

print("Model file found successfully!")

# Initialize MediaPipe Face Landmarker
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

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam!")
    landmarker.close()
    exit()

last_timestamp = 0

while True:

    # Capture frame
    ok, frame = cap.read()

    if not ok:
        print("Failed to capture frame")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, channels = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Generate increasing timestamp
    timestamp = max(
        int(time.monotonic() * 1000),
        last_timestamp + 1
    )
    last_timestamp = timestamp

    # Detect facial landmarks
    results = landmarker.detect_for_video(
        mp_image, timestamp
    )

    # Check for detected face
    if results.face_landmarks:

        face = results.face_landmarks[0]

        # Draw facial landmarks
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

    else:

        cv2.putText(
            frame,
            "No face detected",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Display webcam
    cv2.imshow("Face Mesh Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
landmarker.close()
cv2.destroyAllWindows()