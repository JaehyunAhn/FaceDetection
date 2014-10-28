# -*- coding: utf-8 -*-
"""
    <Deep Belief Network in python>
    - url: http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
    - author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
"""
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

DIR = collect_images('./testImage', False, 0)
image = cv2.imread(DIR[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

test = collect_images('./testImage', True, 'hello')
image_seperation(DIR)
print test['label']

# cv2.imshow("asdf", gray)
# cv2.waitKey(0)