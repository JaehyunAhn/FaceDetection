# -*- coding: utf-8 -*-
from logicLibrary import *
print 'FACE DETECTION SOURCE'
"""
    <Jaehyun Ahn FACE RECOGNITION TEST>
    - Author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
    - Sogang University Datamining Laboratory
    - CV2 is defined in <logicLibrary.py> file

    Reference
     * OpenCV reference: http://goo.gl/Ak5z1u
     * Haar Cascades: http://alereimondo.no-ip.org/OpenCV/34
     * Referred Blog: https://realpython.com/blog/python/face-recognition-with-python/
"""

# loading the images and turn it to gray scale.
DIR = u'./testImage/image3.jpg'
image = cv2.imread(DIR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
FaceDrawing = True
"""
    <EXTRACTION PROCESS>
"""
# Extract faces, eyes, nose, and mouth from <logicLibrary.py>
faces = face_detection(gray, 5)
if len(faces) > 0:
    pass
else:
    exit(-1) # Check number of Faces
cropFace = image
face = faceArea()
# Face Extraction
for (x, y, w, h) in faces:
    if FaceDrawing:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    face.x = x
    face.y = y
    face.w = w
    face.h = h
    cropFace = image[y:y+h, x:x+w]
    # Crop (above) and Normalize (below)
    cropFace = cv2.resize(cropFace, (160, 160))
gray = cv2.cvtColor(cropFace, cv2.COLOR_BGR2GRAY)

# Eye Detection : While() is to find 2 eyes
eye_neighbor = 1
while len(eye_detection(gray, eye_neighbor)) > 2:
    eye_neighbor += 1
    if eye_neighbor >= 50:
        break
eyes = eye_detection(gray, eye_neighbor)
# Nose Detection : While() is to find 1 nose
nose_neighbor = 1
while len(nose_detection(gray, nose_neighbor)) > 1:
    nose_neighbor += 1
    if nose_neighbor >= 50:
        break
nose = nose_detection(gray, nose_neighbor)
# Mouth Detection : While() is to find 1 mouth
mouth_neighbor = 50
while len(mouse_detection(gray, mouth_neighbor)) > 1:
    mouth_neighbor += 1
    if mouth_neighbor >= 400:
        break
mouth = mouse_detection(gray, mouth_neighbor)

"""
    <DRAW A RECTANGLE INTO FACE>
    x,y : location of the rectangle
    w,h: rectangle's width and height
    1. Crop Face y:y+h, x:x+w (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#_=_)
    2. Resize: http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
"""
# initialize
cropNose = cropMouth = cropFace
cropEyes = []
Drawing = False
# Eyes Extraction
for (x, y, w, h) in eyes:
    cropEyes.append(cropFace[y:y+h, x:x+w])
    if Drawing:
        cv2.rectangle(cropFace, (x, y), (x+w, y+h), (0, 255, 0), 1)

# Nose Extraction, HighNose: 콧대까지 측정하기 위한 변수
highNoseHeight = 18
highNoseUnderCut = 10
highNoseWidth = 5
for (x, y, w, h) in nose:
    cropNose = cropFace[y-highNoseHeight:y+h-highNoseUnderCut, x+highNoseWidth:x+w-highNoseWidth]
    if Drawing:
        cv2.rectangle(cropFace, (x+highNoseWidth, y-highNoseHeight), (x+w-highNoseWidth, y+h-highNoseUnderCut), (0, 255, 255), 1)

# Mouth Extraction
for (x, y, w, h) in mouth:
    cropMouth = cropFace[y:y+h, x:x+w]
    if Drawing:
        cv2.rectangle(cropFace, (x, y), (x+w, y+h), (255, 0, 0), 1)

"""
    <CONTOUR PROCESS>
    make a line on cropFace, cropEyes[], cropNose, cropMouth
"""
cropNoseConT = cv2.cvtColor(cropNose, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(cropNoseConT, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# Drawing Process
cv2.drawContours(cropNoseConT, contours, -1, (0, 255, 0), 1)
# print contours

"""
    <REPORTING PROCESS>
"""
# Printout a report
print 'File Name: ' + DIR.split('/')[2]
print "Found {0} faces!".format(len(faces))
print "Found {0} eyes!".format(len(eyes))
print "Found {0} nose!".format(len(nose))
print "Found {0} mouth!".format(len(mouth))
cv2.imshow("Face found", image)
# cv2.imshow("Face", cropFace)
# cv2.imshow("Left Eye", cropEyes[0])
# cv2.imshow("Right Eye", cropEyes[1])
# cv2.imshow("Nose", cropNose)
# cv2.imshow("Mouth", cropMouth)
cv2.waitKey(0)