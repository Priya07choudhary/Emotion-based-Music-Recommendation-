import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Fancy Header
st.markdown(
    "<h1 style='text-align: center; color: #ff6347;'>🎵 Emotion-Based Music Recommender 🎶</h1>",
    unsafe_allow_html=True,
)

# Session state handling
if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Video Processing Class
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Input Section
st.markdown("---")
st.subheader("🎤 Choose Your Preferences")

col1, col2 = st.columns(2)
with col1:
    lang = st.text_input("🎧 Preferred Language")
with col2:
    singer = st.text_input("🎙️ Favorite Singer")

# Start Camera
if lang and singer and st.session_state["run"] != "false":
    st.markdown("---")
    st.info("🎥 Please allow webcam access for emotion detection")
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Display detected emotion (if available)
if emotion:
    st.success(f"✅ Detected Emotion: **{emotion}**")

# Recommend Button
st.markdown("---")
custom_btn = st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
    </style>""", unsafe_allow_html=True)

btn = st.button("🎵 Recommend me songs")

# Button Action
if btn:
    if not emotion:
        st.warning("⚠️ Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        query = f"{lang} {emotion} song {singer}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
