# -*- coding: utf-8 -*-

"""
    <Jaehyun Ahn Face recognition test>
    Reference
     * OpenCV reference: http://goo.gl/Ak5z1u
     * Haar Cascades: http://alereimondo.no-ip.org/OpenCV/34
     * Referred Blog: https://realpython.com/blog/python/face-recognition-with-python/
"""
import cv2

# loading the images and turn it to gray scale.
image = cv2.imread('./testImage/image2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loading the classifiers
faceCascade = cv2.CascadeClassifier('./haarClassifier/haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('./haarClassifier/haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('./haarClassifier/nose.xml')
mouthCascade = cv2.CascadeClassifier('./haarClassifier/mouth.xml')

# image, scaleFactor, neighbors, size, flags return obejct
#   neighbor: defines how many objects are detected near the current one.
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
# minNeighbors = candidates threshold
eyes = eyeCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
# find nose
nose = noseCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
# find mouth
mouth = mouthCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=50,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# print report
print "Found {0} faces!".format(len(faces))
print "Found {0} eyes!".format(len(eyes))
print "Found {0} nose!".format(len(nose))
print "Found {0} mouth!".format(len(mouth))

# Draw a rectangle around the faces
"""
    x,y : location of the rectangle
    w,h: rectangle's width and height
    1. Crop Face y:y+h, x:x+w (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#_=_)
"""
cropFace = image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cropFace = image[y:y+h, x:x+w]

for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

# 콧대까지 측정하기 위한 변수
highNoseThreshold = 20
highNoseUnderCut = 10
for (x, y, w, h) in nose:
    cv2.rectangle(image, (x, y-highNoseThreshold), (x+w, y+h-highNoseUnderCut), (0, 255, 255), 1)

for (x, y, w, h) in mouth:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)

cv2.imshow("Face found", image)
cv2.imshow("Face", cropFace)
cv2.waitKey(0)