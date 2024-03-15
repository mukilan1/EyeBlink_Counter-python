import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Indices of facial landmarks for left and right eye
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize blink counter
blink_count = 0
ear_threshold = 0.25
ear_consecutive_frames = 3
counter = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks
                left_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) 
                                     for i in LEFT_EYE_INDICES], dtype=np.float32)
                right_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) 
                                      for i in RIGHT_EYE_INDICES], dtype=np.float32)

                # Calculate EAR for both eyes
                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0

                # Detect blinks
                if ear < ear_threshold:
                    counter += 1
                else:
                    if counter >= ear_consecutive_frames:
                        blink_count += 1
                    counter = 0

        # Display the blink count
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
