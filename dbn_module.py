# -*- coding: utf-8 -*-
"""
    <Deep Belief Network in python>
    - url: http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
    - author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
"""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from nolearn.dbn import DBN
from dbn_head import *

print "Checking the data..."

"""
    <SEPARATION Eyes, Nose, Mouth>
     - separates target's facial features and save it to different folders
     - collect_images( dictonary, path, attach_label, label_name)
             if attach_label is True,
                return dictionary list of image files
             else
                return list of image files
    - image_separation(DIR)
            read image folder and detect & crop save into separate folder
"""
# image_list = []
# label_index = []
# data_list = {'labels': label_index, 'data': image_list}
# DIR = collect_images(
#     dict=data_list,
#     dir_path='./yeragoData',
#     label_addition=False,
#     label_name=0)
# image = cv2.imread(DIR[0])
# image_separation(DIR)

"""
    <ADD LABEL and CONVERT IMAGES to ARRAY>
    - to use data for Deep, we have to change images to 32 x 32 image size and
        a single line nparray type.
    - data_list = {'data': 3(RGB) * 1024(32 x 32 size) single array's list,
                            'label': array's label }
            (files' length * 3072)
"""
# Itialize data set for line 39.
image_list = []
label_index = []
data_list = {'labels': label_index, 'data': image_list}

# Call collect_images and add a label
i = 0
train = data_list

train = collect_images(
    dict=train,
    dir_path='./yeragoData/faces',
    label_addition=True,
    label_name=0)
train = collect_images(
    dict=train,
    dir_path='./yeragoData/left_eyes',
    label_addition=True,
    label_name=1)
train = collect_images(
    dict=train,
    dir_path='./yeragoData/mouths',
    label_addition=True,
    label_name=2)
train = collect_images(
    dict=train,
    dir_path='./yeragoData/noses',
    label_addition=True,
    label_name=3)

# Save dictionary to .csv file
# save_dictionary(dict=test, filename='testdata.csv')

"""
    <START TRAINING>
    - Referred from : http://goo.gl/GBYZvR
"""
# Collect Data
# dataset = unpickle('data_batch_1')
train['data'] = np.asarray(train['data'])
data_train, labels_train = cvt_tastable_set(train)
data_train = data_train.astype('float') / 255.

# Training Data
n_feat = data_train.shape[1]
n_targets = labels_train.max() + 1
net = DBN(
    [n_feat, n_feat / 3, n_targets],
    epochs=5,
    learn_rates=0.3,
    verbose=4
)
net.fit(data_train, labels_train)

# Test set generation
image_list = []
label_index = []
data_list = {'labels': label_index, 'data': image_list}
test = collect_images(
    dict=data_list,
    dir_path='./yeragoData/testSet/right_eyes',
    label_addition=True,
    label_name=1)
test = collect_images(
    dict=test,
    dir_path='./yeragoData/testSet/left_eyes',
    label_addition=True,
    label_name=1
)
test = collect_images(
    dict=test,
    dir_path='./yeragoData/testSet/mouths',
    label_addition=True,
    label_name=2
)
test = collect_images(
    dict=test,
    dir_path='./yeragoData/testSet/noses',
    label_addition=True,
    label_name=3
)

test['data'] = np.asarray(test['data'])
data_test = test['data'].astype('float') / 255.
labels_test = np.array(test['labels'])
expected = labels_test
predicted = net.predict(data_test)

# Print out Reports
print "Classification report for classifier %s:\n%s\n" % (
    net, classification_report(expected, predicted))
print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)

"""
    <TEST DATA>
    CIFAR-10:
        'data': 10000개, 1개의 array당 3072의 정보, 32x32 1024Red, 1024Green 1024Blue
        'labels': 10000개, 0-9까지의 라벨로 분류되어 있음
"""

"""
# # TEST1 : CIFAR-10
# dict = unpickle('./data_batch_1')
# item = dict['data'][1023]
# test2 = cvt_array_to_BGR(item, 32, 32)
# cv2.imshow("CIFAR-ITEM", test2)
# print type(test2)
# print type(image)
# print isinstance(dict['data'][0], np.ndarray)
# cv2.waitKey(0)

#TEST2 : Handwritten machine learning
dataset = datasets.fetch_mldata("MNIST Original")
asdf =  [[0 for x in xrange(28)] for x in xrange(28)]
for i in range(784):
    row = i/28
    col = i%28
    asdf[row][col] = dataset['data'][0][i]
asdf = np.asarray(asdf)
cv2.imshow("HANDWRITTEN",asdf)
cv2.waitKey(0)

"""