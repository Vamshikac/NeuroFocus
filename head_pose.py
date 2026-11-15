
import cv2
import mediapipe as mp
import numpy as np

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 3D reference points of a face
model_points = np.array([
    (0, 0, 0),
    (0, -330, -65),
    (-225, 170, -135),
    (225, 170, -135),
    (-150, -150, -125),
    (150, -150, -125)
], dtype=np.float64)

# MediaPipe landmark indices
landmark_ids = [1, 152, 33, 263, 61, 291]

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        break

    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        image_points = []

        for idx in landmark_ids:
            landmark = face.landmark[idx]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            image_points.append((x, y))

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        image_points = np.array(
            image_points, dtype=np.float64
        )

        # Camera parameters
        focal_length = width
        center = (width / 2, height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion = np.zeros((4, 1))

        # Calculate head rotation
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            angles = cv2.RQDecomp3x3(rotation_matrix)[0]

            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]

            cv2.putText(
                frame, f"Pitch: {pitch:.2f}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Yaw: {yaw:.2f}",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Roll: {roll:.2f}",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

    cv2.imshow("Head Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
face_mesh.close()
cv2.destroyAllWindows()
