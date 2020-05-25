import os
import numpy as np
import cv2

cascPath = r'C:\Users\cliff\Documents\Work\Projects\face-detection-opencv\project\models\haarcascade_frontalface_default.xml'


def test_run():
    imgPath = r'C:\Users\cliff\Documents\Work\Projects\face-detection-opencv\project\images\face1.jpg'

    face_cascade = cv2.CascadeClassifier(cascPath)

    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_face(byteString):
    face_cascade = cv2.CascadeClassifier(cascPath)

    img = cv2.imdecode(np.fromstring(
        byteString, np.uint8), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    return img