import face_recognition
import cv2
import numpy as np

def load_known_faces(data):
    """Load known face encodings from given paths and IDs."""
    known_face_encodings = []
    known_face_ids = []

    for image_path, face_id in zip(data["images"], data["ids"]):
        try:
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image, num_jitters=10, model='large')[0]
            known_face_encodings.append(face_encoding)
            known_face_ids.append(face_id)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    return known_face_encodings, known_face_ids

def draw_fancy_box(frame, top, right, bottom, left, color=(0, 255, 0), thickness=2):
    """Draw a fancy box around the detected face."""
    length = 30  # Length of box decoration.
    cv2.line(frame, (left, top), (left + length, top), color, thickness)
    cv2.line(frame, (left, top), (left, top + length), color, thickness)

    cv2.line(frame, (right, top), (right - length, top), color, thickness)
    cv2.line(frame, (right, top), (right, top + length), color, thickness)

    cv2.line(frame, (left, bottom), (left + length, bottom), color, thickness)
    cv2.line(frame, (left, bottom), (left, bottom - length), color, thickness)

    cv2.line(frame, (right, bottom), (right - length, bottom), color, thickness)
    cv2.line(frame, (right, bottom), (right, bottom - length), color, thickness)

    return frame
