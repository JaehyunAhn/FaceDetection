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

def collect_images(dir_path, label_detection, label_name):
    images = glob.glob(dir_path + '/*.jpg')
    if label_detection is True:
        image_list = []
        data_list = {'label': label_name, 'data_set': image_list}
        for image_dir in images:
            image = cv2.imread(image_dir)
            image_list.append(image)
        return data_list
    else:
        return images