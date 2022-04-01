import numpy as np
from cv2 import cv2

cap = cv2.VideoCapture(0)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

make_720p()  

while (True):
    #we are capturing the image frame by frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break