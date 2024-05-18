# Face-Recognition-wLive
 
## Real-Time Face Recognition with Mask Detection

This repository contains a Python script that performs real-time face recognition and mask detection using OpenCV and face_recognition libraries. 
The script captures video from your webcam, detects faces, compares them with known faces, and identifies if the detected face is wearing a mask.

* *Real-time Face Detection: Continuously captures video from the webcam and detects faces in the video feed.
* Face Recognition: Compares detected faces with a set of known faces to identify individuals.
* Mask Detection: Identifies if a detected face is wearing a mask and displays a mask indicator.
* Confidence Score: Displays a confidence score for the recognized faces.

Prerequisites
⚹Python 3.x
⚹OpenCV
⚹face_recognition
⚹numpy

You can install the required packages using pip:

```
pip install opencv-python face_recognition numpy
```

## How to Use

Add Images of Known Faces: Place images of known individuals in a directory and update the data dictionary with the paths to these images and corresponding IDs.


```ruby

data = {
    "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
    "ids": ["Person1", "Person2"]
}

```

Run the Script: Execute the script to start the video capture and face recognition process.

```
 python face_recognition_mask_detection.py

```

Interact with the Application: The script will display the video feed with rectangles around detected faces, indicating recognized individuals and mask status. Press 'q' to quit the application.