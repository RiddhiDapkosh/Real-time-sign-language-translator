import streamlit as st
import cv2

from utils.database import register_user, login_user, save_history, get_history
from utils.preprocessing import extract_landmarks

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = ""

if "run" not in st.session_state:
    st.session_state.run = False

if "sentence" not in st.session_state:
    st.session_state.sentence = []

# =========================
# TRANSLATION (SAFE LOCAL)
# =========================
def translate(sentence):
    sentence = sentence.lower()

    hindi = sentence.replace("hello", "नमस्ते").replace("i", "मैं").replace("am", "हूँ")
    marathi = sentence.replace("hello", "नमस्कार").replace("i", "मी").replace("am", "आहे")

    return {
        "english": sentence,
        "hindi": hindi,
        "marathi": marathi
    }

# =========================
# SIMPLE SIGN PREDICTOR
# =========================
def predict_sign(landmarks):
    if landmarks is None:
        return None
    return "Hello"

# =========================
# LOGIN PAGE
# =========================
if not st.session_state.logged_in:
    st.title("🤟 ASL Translator App")

    mode = st.radio("Choose", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Account created!")
            else:
                st.error("User already exists")

    if mode == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid login")

# =========================
# MAIN APP
# =========================
else:
    st.sidebar.title(f"Welcome {st.session_state.user}")
    page = st.sidebar.radio("Menu", ["Live Camera", "History", "Logout"])

    if page == "Logout":
        st.session_state.logged_in = False
        st.session_state.user = ""
        st.rerun()

    # =========================
    # HISTORY
    # =========================
    if page == "History":
        st.subheader("📜 History")

        for row in get_history(st.session_state.user):
            st.write(row)

    # =========================
    # CAMERA
    # =========================
    if page == "Live Camera":
        st.subheader("📷 Real-Time ASL Detection")

        start = st.button("Start Camera")
        stop = st.button("Stop Camera")
        snap = st.button("Snapshot")

        frame_box = st.image([])

        cap = None

        if start:
            st.session_state.run = True
            cap = cv2.VideoCapture(0)

        if stop:
            st.session_state.run = False

        while st.session_state.run:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # SAFE MediaPipe call (after fixing install)
            frame, landmarks = extract_landmarks(frame)

            sign = predict_sign(landmarks)

            if sign:
                st.session_state.sentence.append(sign)

            cv2.putText(frame, str(sign), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            frame_box.image(frame)

            if snap:
                sentence = " ".join(st.session_state.sentence)

                result = translate(sentence)

                save_history(st.session_state.user, sentence)

                st.success("Saved!")
                st.write(result)

                st.session_state.sentence = []

            if not st.session_state.run:
                break

        if cap:
            cap.release()