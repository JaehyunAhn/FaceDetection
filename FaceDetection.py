# -*- coding: utf-8 -*-
from logicLibrary import *
print 'Image Detection source'
"""
    <Jaehyun Ahn Face recognition test>
    - Author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
    - Sogang University Datamining Laboratory
    - CV2 is defined in <logicLibrary.py> file

    Reference
     * OpenCV reference: http://goo.gl/Ak5z1u
     * Haar Cascades: http://alereimondo.no-ip.org/OpenCV/34
     * Referred Blog: https://realpython.com/blog/python/face-recognition-with-python/
"""

# loading the images and turn it to gray scale.
image = cv2.imread('./testImage/image7.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract faces, eyes, nose, and mouth from <logicLibrary.py>
faces = face_detection(gray, 5)
cropFace = image
face = faceArea()
# Face Extraction
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face.x = x
    face.y = y
    face.w = w
    face.h = h
    cropFace = image[y:y+h, x:x+w]
    # Crop (above) and Normalize (below)
    cropFace = cv2.resize(cropFace, (160, 160))
gray = cv2.cvtColor(cropFace, cv2.COLOR_BGR2GRAY)
# Check length of Faces
if len(faces) > 0:
    pass
else:
    exit(-1)

# Eye Detection : While() is to find 2 eyes
eye_neighbor = 1
while len(eye_detection(gray, eye_neighbor)) > 2:
    eye_neighbor += 1
    if eye_neighbor >= 50:
        break
eyes = eye_detection(gray, eye_neighbor)
# Nose Detection
nose = nose_detection(gray, 5)
# Mouth Detection : While() is to find 1 mouth
mouth_neighbor = 50
while len(mouse_detection(gray, mouth_neighbor)) > 1:
    mouth_neighbor += 1
    if mouth_neighbor >= 400:
        break
mouth = mouse_detection(gray, mouth_neighbor)

# Draw a rectangle around the faces
"""
    x,y : location of the rectangle
    w,h: rectangle's width and height
    1. Crop Face y:y+h, x:x+w (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#_=_)
    2. Resize: http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
"""
# 눈추출
for (x, y, w, h) in eyes:
    cv2.rectangle(cropFace, (x, y), (x+w, y+h), (0, 255, 0), 1)

# 코추출: 콧대까지 측정하기 위한 변수
highNoseHeight = 18
highNoseUnderCut = 10
highNoseWidth = 5
for (x, y, w, h) in nose:
    cv2.rectangle(cropFace, (x+highNoseWidth, y-highNoseHeight), (x+w-highNoseWidth, y+h-highNoseUnderCut), (0, 255, 255), 1)

# 입추출
for (x, y, w, h) in mouth:
    cv2.rectangle(cropFace, (x, y), (x+w, y+h), (255, 0, 0), 1)

# print report
print "Found {0} faces!".format(len(faces))
print "Found {0} eyes!".format(len(eyes))
print "Found {0} nose!".format(len(nose))
print "Found {0} mouth!".format(len(mouth))
cv2.imshow("Face found", image)
cv2.imshow("Face", cropFace)
cv2.waitKey(0)