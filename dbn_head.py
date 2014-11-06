# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import glob
import shutil
import csv
from logicLibrary import *

# open data and translate it as dictionary data type
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# collect images <.jpg>file, in dir_path, and attach label if user wants
def collect_images(dict, dir_path, label_addition, label_name):
    images = glob.glob(dir_path + '/*.jpg')
    # add a new label and make a dictionary type list
    if label_addition is True:
        for image_dir in images:
            image = cv2.imread(image_dir)
            array = cvt_BGR_to_array(image, 32, 32)
            array = np.asarray(array)
            dict['data'].append(array)
            dict['labels'].append(label_name)
        return dict
    # return list of <.jpg> file list
    else:
        return images

# read images in the folder and seperate images to eyes, noses, mouth
def image_separation(image_list):
    print '[Notice] Image_separation function called.'
    for image_path in image_list:
        print 'PROCESSING: ', image_path
        dir_path = './'
        dir_path += (image_path.split('/')[1] + '/')
        file_name = image_path.split('/')[2][:-4]
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
        cv2.imwrite(dir_path + file_name + '_face.jpg', cropFace)

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
        if len(cropEyes) == 1:
            print "Only one eye detected in ", str(file_name)
            cv2.imwrite(dir_path + file_name + '_left_eye.jpg', cropEyes[0])
        elif len(cropEyes) >= 2:
            cv2.imwrite(dir_path + file_name + '_left_eye.jpg', cropEyes[0])
            cv2.imwrite(dir_path + file_name + '_right_eye.jpg', cropEyes[1])
        else:
            print "There's no Eye in ", str(file_name)

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
        cv2.imwrite(dir_path + file_name + '_nose.jpg', cropNose)

        # Mouth Extraction
        for (x, y, w, h) in mouth:
            cropMouth = cropFace[y:y+h, x:x+w]
            if Drawing:
                cv2.rectangle(cropFace, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.imwrite(dir_path + file_name + '_mouth.jpg', cropMouth)
        print "    Found {0} faces!".format(len(faces))
        print "    Found {0} eyes!".format(len(eyes))
        print "    Found {0} nose!".format(len(nose))
        print "    Found {0} mouth!".format(len(mouth))
    print "Done."
    dir_path = image_list[0].split('/')[1]
    move_items_to_folders(dir_path)
    return 1

def move_items_to_folders(dir_path):
    # grap images
    faces = glob.glob(dir_path+'/*_face.jpg')
    noses = glob.glob(dir_path+'/*_nose.jpg')
    mouths = glob.glob(dir_path+'/*_mouth.jpg')
    left_eyes = glob.glob(dir_path+'/*_left_eye.jpg')
    right_eyes = glob.glob(dir_path+'/*_right_eye.jpg')
    # move it to suitable folders
    for face in faces:
        try:
            shutil.move(face, dir_path+'/faces/')
        except:
            print '[ERROR dbn_head.py] Failed to move faces to folder'
        else:
            print len(faces), 'faces were moved'
    # Move Nose
    for nose in noses:
        try:
            shutil.move(nose, dir_path+'/noses/')
        except:
            print '[ERROR dbn_head.py] Failed to move noses to folder'
        else:
            print len(noses), 'noses were moved'
    # Move mouth
    for mouth in mouths:
        try:
            shutil.move(mouth, dir_path+'/mouths/')
        except:
            print '[ERROR dbn_head.py] Failed to move mouths to folder'
        else:
            print len(mouths), 'mouses were moved'
    # Left eyes
    for left_eye in left_eyes:
        try:
            shutil.move(left_eye, dir_path+'/left_eyes/')
        except:
            print '[ERROR dbn_head.py] Failed to move left_eyes to folder'
        else:
            print len(left_eyes), 'left eyes were moved'
    # Right eyes
    for right_eye in right_eyes:
        try:
            shutil.move(right_eye, dir_path+'/right_eyes/')
        except:
            print '[ERROR dbn_head.py] Failed to move right_eyes to folder'
        else:
            print len(right_eyes), 'right_eyes were moved'

def cvt_array_to_BGR(array, width, height):
    # BGR
    image = [[0 for x in xrange(width)] for x in xrange(height)]
    for row in range(width):
        for col in range(height):
            image[row][col] = [0, 0, 0]
    for i in range(1024):
        row = int(i / width)
        col = int(i % width)
        image[row][col] = [array[i+2048], array[i+1024], array[i]]
    # convert numpy array
    image = np.asarray(image)
    return image

def cvt_BGR_to_array(BGR, width, height):
    # BGR to RGB array
    array = []
    crop_image = cv2.resize(BGR, (width, height))
    # Save Red Color
    for row in range(width):
        for col in range(height):
            array.append(crop_image[row][col][2])
    # Save Blue Color
    for row in range(width):
        for col in range(height):
            array.append(crop_image[row][col][1])
    # Save Green Color
    for row in range(width):
        for col in range(height):
            array.append(crop_image[row][col][0])
    return array

# Save Dictonary to <filename.csv> file
def save_dictionary(dict, filename):
    with open(filename, 'wb') as csv_file:
        w = csv.DictWriter(csv_file, dict.keys())
        w.writeheader()
        w.writerow(dict)
        print filename, 'was written!'

# convert it to java dictonary type
def cvt_tastable_set(dict):
    data_train = np.vstack([dict['data']])
    labels_train = np.hstack([dict['labels']])
    return data_train, labels_train
