# -*- coding: utf-8 -*-
"""
    <Deep Belief Network in python>
    - url: http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
    - author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
"""
__author__ = 'MLLAB'
from dbn_head import *

print "Checking the data..."
"""
    <COLLECT IMAGES>
    - function: collect_images( path, attach_label, label_name)
                    if attach_label is True,
                        return dictionary list of image files
                    else
                        return list of image files
"""
DIR = u'./testImage/image2.jpg'
image = cv2.imread(DIR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# crop_size = (32, 32)
# result = cv2.resize(gray, crop_size)

DIR = collect_images('./testImage', False, 0)
image = cv2.imread(DIR[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

test = collect_images('./testImage', True, 'hello')
print test['label']

cv2.imshow("asdf", gray)
cv2.waitKey(0)