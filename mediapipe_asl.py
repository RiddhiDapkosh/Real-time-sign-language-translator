import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# SIMPLE SIGN CLASSIFIER (RULE-BASED DEMO)
# Replace this with ML model later
# =========================
def classify_sign(landmarks):
    """
    Very simple heuristic demo classifier.
    Replace with trained ML model for real ASL.
    """

    # Thumb tip (4), Index tip (8)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Open/close fingers check
    fingers_up = 0
    tips = [8, 12, 16, 20]

    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers_up += 1

    # RULES (very basic demo)
    if fingers_up == 0:
        return "Fist (A)"
    elif fingers_up == 5:
        return "Open Palm (B)"
    elif fingers_up == 1:
        return "Pointing (D)"
    elif fingers_up == 2:
        return "Peace (V)"
    else:
        return "Unknown"

# =========================
# CAMERA START
# =========================
cap = cv2.VideoCapture(0)

history = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    sign_text = "No Hand Detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Convert landmarks to numpy
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm)

            sign_text = classify_sign(landmarks)
            history.append(sign_text)

    # Smooth prediction (majority vote)
    if len(history) > 0:
        sign_text = max(set(history), key=list(history).count)

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f"Sign: {sign_text}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("ASL MediaPipe Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()