from pathlib import Path
import cv2
import mediapipe as mp
import time


# --------------------------------------------------
# Hand landmark connections
# --------------------------------------------------

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# --------------------------------------------------
# 1. MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Path to the MediaPipe hand landmarker model
MODEL_PATH = str(
    Path(__file__).parent / "models" / "hand_landmarker.task"
)

print("Model path:", MODEL_PATH)
print("Model exists:", Path(MODEL_PATH).exists())


# Configure Hand Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# Create hand landmarker
landmarker = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# 2. Start webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


# --------------------------------------------------
# 3. Process webcam frames
# --------------------------------------------------

frame_timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break


    # OpenCV gives BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp must increase for VIDEO mode
    frame_timestamp += 1


    # Detect hands
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # --------------------------------------------------
    # 4. Draw landmarks and connections
    # --------------------------------------------------

    if result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            # ------------------------------------------
            # Convert all 21 landmarks to pixel points
            # ------------------------------------------

            points = []

            for landmark in hand_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                points.append((x, y))


            # ------------------------------------------
            # Draw connections
            # ------------------------------------------

            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------
            # Draw landmark points and numbers
            # ------------------------------------------

            for landmark_id, point in enumerate(points):

                x, y = point


                # Draw green dot
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


                # Draw landmark number
                cv2.putText(
                    frame,
                    str(landmark_id),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )


    # --------------------------------------------------
    # 5. Display
    # --------------------------------------------------

    cv2.imshow(
        "ISL - MediaPipe Hand Landmarks",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 6. Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Program stopped.")