# -*- coding: utf-8 -*-
import cv2
# loading the classifiers
faceCascade = cv2.CascadeClassifier('./haarClassifier/haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('./haarClassifier/haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('./haarClassifier/nose.xml')
mouthCascade = cv2.CascadeClassifier('./haarClassifier/mouth.xml')

# Face Detection
# image, scaleFactor, neighbors, size, flags return obejct
#   neighbor: defines how many objects are detected near the current one.
def face_detection(image, neighbors):
    item = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=neighbors,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return item

# 얼굴을 찾으면 얼굴 범위가 저장되는 클래스
class faceArea:
    def __init__(self):
        x = 0
        y = 0
        w = 0
        h = 0

# minNeighbors = candidates threshold
def eye_detection(image, neighbors):
    item = eyeCascade.detectMultiScale(
        image,
        scaleFactor=1.5,
        minNeighbors=neighbors,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return item

# Find Nose
def nose_detection(image, neighbors):
    item = noseCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=neighbors,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return item

# Find Mouth
def mouse_detection(image, neighbors):
    item = mouthCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=neighbors,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return item