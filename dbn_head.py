# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cPickle
import glob
from logicLibrary import *

# open data and translate it as dictionary data type
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# collect images <.jpg>file, in dir_path, and attach label if user wants
def collect_images(dir_path, label_detection, label_name):
    images = glob.glob(dir_path + '/*.jpg')
    # add a new label and make a dictionary type list
    if label_detection is True:
        image_list = []
        data_list = {'label': label_name, 'data_set': image_list}
        for image_dir in images:
            image = cv2.imread(image_dir)
            image_list.append(image)
        return data_list
    # return list of <.jpg> file list
    else:
        return images

# read images in the folder and seperate images to eyes, noses, mouth
def image_seperation(image_list):
    for image_path in image_list:
        print 'PROCESSING: ', image_path
        file_name = image_path.split('/')[1][:-3]
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        FaceDrawing = True
        cropSize = (200, 200)
        """
            <EXTRACTION PROCESS>
        """
        # Extract faces, eyes, nose, and mouth from <logicLibrary.py>
        faces = face_detection(gray, 5)
        if len(faces) <= 0:
            print 'ERROR: THERE IS NO FACE!'
            return -1
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
            cropFace = cv2.resize(cropFace, cropSize)
        gray = cv2.cvtColor(cropFace, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(file_name + '_face.jpg', cropFace)

        # Eye Detection : While() is to find 2 eyes
        eye_neighbor = 1
        while len(eye_detection(gray, eye_neighbor)) > 2:
            eye_neighbor += 1
            if eye_neighbor >= 100:
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
        cv2.imwrite(file_name + '_left_eye.jpg', cropEyes[0])
        if len(cropEyes) >= 2:
            cv2.imwrite(file_name + '_right_eye.jpg', cropEyes[1])

        # Nose Extraction, HighNose: 콧대까지 측정하기 위한 변수
        highNoseHeight = int(cropSize[0] * 0.14)
        highNoseUnderCut = int(cropSize[0] * 0.05)
        highNoseWidthCut = int(cropSize[0] * 0.06)
        for (x, y, w, h) in nose:
            cropNose = cropFace[y-highNoseHeight:y+h-highNoseUnderCut,
                       x+highNoseWidthCut:x+w-highNoseWidthCut]
            if Drawing:
                cv2.rectangle(cropFace, (x+highNoseWidthCut, y-highNoseHeight),
                              (x+w-highNoseWidthCut, y+h-highNoseUnderCut), (0, 255, 255), 1)
        cv2.imwrite(file_name + '_nose.jpg', cropNose)

        # Mouth Extraction
        for (x, y, w, h) in mouth:
            cropMouth = cropFace[y:y+h, x:x+w]
            if Drawing:
                cv2.rectangle(cropFace, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.imwrite(file_name + '_mouth.jpg', cropMouth)
        print "    Found {0} faces!".format(len(faces))
        print "    Found {0} eyes!".format(len(eyes))
        print "    Found {0} nose!".format(len(nose))
        print "    Found {0} mouth!".format(len(mouth))
    print "Done."