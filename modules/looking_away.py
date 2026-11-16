
import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model_points = np.array([
    (0, 0, 0),
    (0, -330, -65),
    (-225, 170, -135),
    (225, 170, -135),
    (-150, -150, -125),
    (150, -150, -125)
], dtype=np.float64)

landmark_ids = [1, 152, 33, 263, 61, 291]

baseline = None
samples = []
calibrating = False
sample_count = 30

away_start = None
away_threshold = 1.5

YAW_LIMIT = 20
PITCH_LIMIT = 15

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]
        image_points = []

        for idx in landmark_ids:
            landmark = face.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            image_points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        image_points = np.array(image_points, dtype=np.float64)

        focal_length = width
        center = (width / 2, height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = cv2.RQDecomp3x3(rotation_matrix)[0]

            pitch, yaw, roll = angles
            current = np.array([pitch, yaw, roll])

            if calibrating:
                samples.append(current)

                if len(samples) >= sample_count:
                    baseline = np.mean(samples, axis=0)
                    calibrating = False
                    samples = []
                    away_start = None

            if baseline is None:

                message = "Calibrating..." if calibrating else "Press C to calibrate"

                cv2.putText(
                    frame, message, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2
                )

                if calibrating:
                    cv2.putText(
                        frame, f"Frames: {len(samples)}/{sample_count}",
                        (30, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2
                    )

            else:

                relative = current - baseline

                pitch_change = relative[0]
                yaw_change = relative[1]

                looking_away = (
                    abs(yaw_change) > YAW_LIMIT or
                    abs(pitch_change) > PITCH_LIMIT
                )

                if looking_away:

                    if away_start is None:
                        away_start = time.time()

                    duration = time.time() - away_start

                    if duration >= away_threshold:
                        status = "LOOKING AWAY"
                        color = (0, 0, 255)
                    else:
                        status = "HEAD TURN DETECTED"
                        color = (0, 255, 255)

                else:
                    away_start = None
                    status = "LOOKING AT SCREEN"
                    color = (0, 255, 0)

                cv2.putText(
                    frame, status, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2
                )

                cv2.putText(
                    frame, f"Yaw: {yaw_change:.2f}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, f"Pitch: {pitch_change:.2f}",
                    (30, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

    else:
        away_start = None
        cv2.putText(
            frame, "NO FACE DETECTED", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2
        )

    cv2.imshow("Looking Away Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and not calibrating:
        baseline = None
        samples = []
        calibrating = True
        away_start = None

    elif key == ord('q'):
        break

camera.release()
face_mesh.close()
cv2.destroyAllWindows()