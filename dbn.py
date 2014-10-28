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
    - collect_images( dictonary, path, attach_label, label_name)
             if attach_label is True,
                return dictionary list of image files
             else
                return list of image files
    - image_separation(DIR)
            read image folder and detect & crop save into separate folder
"""
# initialize data set
image_list = []
label_index = []
data_list = {'label': label_index, 'data_set': image_list}

DIR = collect_images(data_list, './testImage', False, 0)
image = cv2.imread(DIR[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

test = collect_images(data_list, './testImage', True, 'hello')
# print test['data_set']
# image_separation(DIR)

"""
    <TEST DATA>
    CIFAR-10:
        'data': 10000개, 1개의 array당 3072의 정보, 32x32 1024Red, 1024Green 1024Blue
        'labels': 10000개, 0-9까지의 라벨로 분류되어 있음
"""
# TEST1 : CIFAR-10
# dict = unpickle('./data_batch_1')
# item = dict['data'][1023]
# test2 = cvt_array_to_BGR(item, 32, 32)
# cv2.imshow("ssss", test2)
# print type(test2)
# print type(image)
# print isinstance(dict['data'][0], np.ndarray)
# cv2.waitKey(0)

#TEST2 : Handwritten machine learning
dataset = datasets.fetch_mldata("MNIST Original")
# asdf =  [[0 for x in xrange(28)] for x in xrange(28)]
# for i in range(784):
#     row = i/28
#     col = i%28
#     asdf[row][col] = dataset['data'][0][i]
# asdf = np.asarray(asdf)
# cv2.imshow("asdfgg",asdf)
# cv2.waitKey(0)
print dataset