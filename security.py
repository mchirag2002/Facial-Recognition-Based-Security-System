import face_recognition
from cv2 import cv2
import numpy as np
import pyfirmata
from pyfirmata import Arduino
import time
import os

board = Arduino('COM3')
face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
video_capture = cv2.VideoCapture(0)


path = 'E:\PROJECTS\FACIAL RECOGNITION\Database'
peopleImg = []
known_face_names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')  # politicians image/Amit_Shah.jpg
    peopleImg.append(curimg)
    # Here 0 depicts the first part of image name
    known_face_names.append(os.path.splitext(cl)[0])
# print(peopleName)


def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings


known_face_encodings = findEncoding(peopleImg)


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name == "Unknown":
                board.digital[13].write(1)
                time.sleep(2)
                board.digital[13].write(0)
                time.sleep(1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+h]
                    img_item = "unknown person.png"
                    cv2.imwrite(img_item, roi_gray)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
