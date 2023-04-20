import cv2 as cv
import sys
import os
import numpy as np
from time import sleep
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, OUT)
pwm = GPIO.PWM(3,50)
pwm.start(0)

facecam = PiCamera()
facecam.rotation = 270
facecam.resolution = (144,112)
facecam.framerate = 32
rawCapture = PiRGBArray(facecam, size=(144,112))
face_datasets = '/Users/sidarthraman/Documents/sortingcode/facedatasets'

def SetAngle(angle):
    #formula for angle
    duty = angle/18+2
    GPIO.output(3, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(3, False)
    pwm.ChangeDutyCycle(90)

(images, labels, names, id) = ([], [], {}, 0)
for (subdirectories, directories, files) in os.walk(face_datasets):
    for subdirectories in directories:
        names[id] = subdirectories
        person_path = os.path.join(face_datasets, subdirectories)
        for imagename in os.listdir(person_path):
            path = person_path + '/' + imagename
            label = id
            images.append(cv.imread(path, 0))
            labels.append(int(label))
        id+=1
(images, labels) = [np.array(imagelis) for imagelis in [images,labels]]

#creates the model
model = cv.face.LBPHFaceRecognizer_create()
model.train(images, labels)
face_cascade = cv.CascadeClassifier('/Users/sidarthraman/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
while (facecam.isOpened()):
    ret, frame = facecam.read()
    new_grayscale_face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(new_grayscale_face)
    SetAngle(0)
    for (column, row, width, height) in detected_faces:
        face_resize = cv.resize(new_grayscale_face[row: row + height, column: column + width], (130,100))
        face_prediction = model.predict(face_resize)
        cv.rectangle(frame, (column, row), (column + width, row + height), (255,0,0), 2)
        if (face_prediction[1] < 85):
            #delete putText
            SetAngle(90)
        else:
            SetAngle(0)

        cv.imshow('facecamdetect', frame)

        key = cv.waitKey(ord('q'))
        if (key == 'q'):
            break

facecam.release()
cv.destroyAllWindows()
