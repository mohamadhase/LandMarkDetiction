import cv2
import numpy as np

from utils import sunglasses_filter

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the video stream
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # Read the frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around each face and display the image
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('stream', frame)
        cv2.imshow('Face ', frame[y:y+h, x:x+w])
        cv2.imwrite("filename.jpg",  frame[y:y+h, x:x+w])
        img = cv2.imread(sunglasses_filter("filename.jpg"))
        cv2.imshow('Image', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
