import cv2
import face_recognition
import streamlit as st
import numpy as np
from gtts import gTTS
import base64
from PIL import Image
import os

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="waiiiiiittttingggmaster99",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Session state for encodings
# =========================
if "known_encodings" not in st.session_state:
    # Load defaults only once
    doyun_encode = face_recognition.face_encodings(
        face_recognition.load_image_file("doyu.jpg")
    )[0]
    degii_encode = face_recognition.face_encodings(
        face_recognition.load_image_file("degii.jpg")
    )[0]
    st.session_state.known_encodings = [doyun_encode, degii_encode]
    st.session_state.known_names = ["Doyoun", "Degi"]

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Controls")
    stop = st.button("⏹️ Stop Webcam")
    restart = st.button("🔄 Restart Webcam")
    st.markdown("---")

    st.subheader("📤 Upload Face Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    person_name = st.text_input("Enter a name for this person:")
    save_btn = st.button("💾 Save & Add to Known Faces")

    if save_btn and uploaded_file and person_name.strip() != "":
        # Open and resize image
        image = Image.open(uploaded_file)
        image = image.resize((400, 500))  # Resize to 400x500 px

        # Save temporarily inside Streamlit cloud (/tmp/)
        save_filename = os.path.join("/tmp", f"{person_name}.jpg")
        image.save(save_filename)

        # Encode face and add to session state
        img_loaded = face_recognition.load_image_file(save_filename)
        encodings = face_recognition.face_encodings(img_loaded)

        if len(encodings) > 0:
            st.session_state.known_encodings.append(encodings[0])
            st.session_state.known_names.append(person_name)
            st.success(f"✅ {person_name} added and saved temporarily.")
        else:
            st.error("⚠️ No face detected in the uploaded image.")

    elif save_btn and not uploaded_file:
        st.error("⚠️ Please upload an image before saving.")
    elif save_btn and person_name.strip() == "":
        st.error("⚠️ Please enter a name before saving.")

# =========================
# Audio Helper
# =========================
tts = gTTS('Degi', lang='en')
tts.save("deg.mp3")

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# =========================
# Webcam Runner
# =========================
frame_placeholder = st.empty()

def run_webcam():
    webcam = cv2.VideoCapture(0)

    while webcam.isOpened() and not stop:
        ret, frame = webcam.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locs)

        for (top, right, bottom, left), face_encoded in zip(face_locs, face_encodings):
            distances = face_recognition.face_distance(st.session_state.known_encodings, face_encoded)
            best_match_index = np.argmin(distances)

            if distances[best_match_index] < 0.4:
                name = st.session_state.known_names[best_match_index]
            else:
                name = "Unknown"

            if name == "Degi":
                autoplay_audio("deg.mp3")

            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            y = top - 10 if top - 10 > 10 else top + 10
            cv2.putText(
                rgb_frame,
                name,
                (left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        frame_placeholder.image(rgb_frame, channels="RGB")

    webcam.release()

# =========================
# Start webcam
# =========================
if not stop:
    run_webcam()
elif restart:
    run_webcam()
