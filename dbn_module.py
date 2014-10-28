# -*- coding: utf-8 -*-
"""
    <Deep Belief Network in python>
    - url: http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
    - author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
"""
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
# DIR = collect_images(
#     dict=data_list,
#     dir_path='./testImage',
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
data_list = {'label': label_index, 'data': image_list}

# Call collect_images and add a label
test = collect_images(
    dict=data_list,
    dir_path='./testImage/faces',
    label_addition=True,
    label_name=0)
test = collect_images(
    dict=test,
    dir_path='./testImage/left_eyes',
    label_addition=True,
    label_name=1)
test = collect_images(
    dict=test,
    dir_path='./testImage/mouths',
    label_addition=True,
    label_name=2)
test = collect_images(
    dict=test,
    dir_path='./testImage/noses',
    label_addition=True,
    label_name=3)
test = collect_images(
    dict=test,
    dir_path='./testImage/right_eyes',
    label_addition=True,
    label_name=4)

print len(test['data'])

java_array = cvt_java_array(test)
# Save dictionary to .csv file
# save_dictionary(dict=test, filename='testdata.csv')

"""
    <START TRAINING>
    - Referred from : http://goo.gl/GBYZvR
"""
dataset = datasets.fetch_mldata("MNIST Original")
print type(dataset), type(dataset.data), type(dataset.data[0]), type(dataset.target), type(dataset.target[0])
print type(java_array), type(java_array.data), type(java_array.data[0]), type(java_array.label), type(java_array.label[0])
# scale the data to the range [0, 1] and then construct the training
# and testing splits
# version 1 == Jaehyun's, version 2 == tests'
version = 5
if version == 5:
    (trainX, testX, trainY, testY) = train_test_split(
        java_array.data / 255.0, java_array.label.astype('int0'), test_size=0.33)
elif version == 10:
    (trainX, testX, trainY, testY) = train_test_split(
        dataset.data / 255.0, dataset.target.astype('int0'), test_size=0.33)

print trainX.shape
# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)
dbn = DBN(
    [trainX.shape[1], 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=version,
    verbose=1)
dbn.fit(trainX, trainY)
# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
print classification_report(testY, preds)
# randomly select a few of the test instances
for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
    # classify the digit
    pred = dbn.predict(np.atleast_2d(testX[i]))

    # reshape the feature vector to be a 28x28 pixel image, then change
    # the data type to be an unsigned 8-bit integer
    image = (testX[i] * 255).reshape((28, 28)).astype("uint8")

    # show the image and prediction
    print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
    cv2.imshow("Digit", image)
    cv2.waitKey(0)

"""
    <TEST DATA>
    CIFAR-10:
        'data': 10000개, 1개의 array당 3072의 정보, 32x32 1024Red, 1024Green 1024Blue
        'labels': 10000개, 0-9까지의 라벨로 분류되어 있음
"""
"""
# TEST1 : CIFAR-10
dict = unpickle('./data_batch_1')
item = dict['data'][1023]
test2 = cvt_array_to_BGR(item, 32, 32)
cv2.imshow("CIFAR-ITEM", test2)
print type(test2)
print type(image)
print isinstance(dict['data'][0], np.ndarray)
cv2.waitKey(0)

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