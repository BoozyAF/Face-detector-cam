import time
import datetime
import sys
import cv2

#Library location
sys.path.append('/usr/local/lib/python3.9/site-packages')

#Selected camera device
cap = cv2.VideoCapture(0)

#Face finder
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

#Shows the camera POV with colored box around the detected face and the body
while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 6)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

    #for (x, y, width, height) in bodies:
    #    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    cv2.imshow("Camera", frame)

    #Press "q" to kill the task
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()