import cv2
import face_recognition
import time
import numpy as np

data = {
    "images": ["image/path"],
    "ids": ["Zanlı"]
}

known_face_encodings = []
known_face_ids = []

for image_path, face_id in zip(data["images"], data["ids"]):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image, num_jitters=10, model='large')[0]
    known_face_encodings.append(face_encoding)
    known_face_ids.append(face_id)

video_capture = cv2.VideoCapture(0)
while True:

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, faces)

    for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_id = "Unknown"
        face_confidence = 0.0

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        
        if matches[best_match_index]:
            face_id = known_face_ids[best_match_index]
            scaling_factor = 1.3
            face_confidence = scaling_factor / (1.0 + scaling_factor * face_distances[best_match_index])

        if face_confidence < 0.5:
            landmarks = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])[0]
            mask_top = min(landmarks["nose_bridge"], key=lambda p: p[1])[1]
            cv2.rectangle(frame, (left, mask_top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Maske", (left, top - 10),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 2)
            continue
        
        t = 5
        l = 30
        rt = 1
        
        top1 = top + bottom
        bottom1 = left + right


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), rt)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), rt)

        cv2.line(frame, (left, top), (left + l, top), (0, 255, 0), t) 
        cv2.line(frame, (left, top), (left, top + l), (0, 255, 0), t)  

        cv2.line(frame, (right, top), (right - l, top), (0, 255, 0), t)  
        cv2.line(frame, (right, top), (right, top + l), (0, 255, 0), t) 

        cv2.line(frame, (left, bottom), (left + l, bottom), (0, 255, 0), t) 
        cv2.line(frame, (left, bottom), (left, bottom - l), (0, 255, 0), t) 

        cv2.line(frame, (right, bottom), (right - l, bottom), (0, 255, 0), t)
        cv2.line(frame, (right, bottom), (right, bottom - l), (0, 255, 0), t)  
                
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{face_id} ({face_confidence:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
