# src/face_recognition_app.py

import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
from face_utils import load_known_faces, draw_fancy_box

# Load known faces (adjust paths as needed)
data = {
    "images": ["assets/sample_face.jpg"],  # Replace with actual paths
    "ids": ["Suspect"]
}

known_face_encodings, known_face_ids = load_known_faces(data)

# Streamlit Interface Elements
st.title("Face Recognition System")
st.sidebar.title("Configuration")
st.sidebar.subheader("Video Settings")

video_source = st.sidebar.selectbox("Select Video Source", options=["Webcam", "Upload Video"])
show_fancy_box = st.sidebar.checkbox("Show Fancy Box", value=True)

# Upload video option (for demonstration or testing)
uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Start the video feed if the user selects webcam or uploads a video
run = st.button("Start Face Recognition")

if run:
    stframe = st.empty()  # Placeholder for displaying video frames

    # Initialize video capture
    if video_source == "Webcam":
        video_capture = cv2.VideoCapture(0)  # Use webcam
    else:
        video_capture = cv2.VideoCapture(uploaded_file.name)  # Use uploaded file path

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces
        faces = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, faces)

        for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_id = "Unknown"
            face_confidence = 0.0

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                face_id = known_face_ids[best_match_index]
                face_confidence = 1 - face_distances[best_match_index]

            # Draw fancy box and display name if enabled
            if show_fancy_box:
                frame = draw_fancy_box(frame, top * 4, right * 4, bottom * 4, left * 4)
            cv2.putText(frame, f"{face_id} ({face_confidence:.2f})", (left * 4, top * 4 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display frame in Streamlit
        stframe.image(frame, channels="BGR")

        if st.sidebar.button("Stop"):
            break

    # Release video capture and close Streamlit app
    video_capture.release()
    cv2.destroyAllWindows()
