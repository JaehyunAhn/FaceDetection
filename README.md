# Face Detection Algorithm
* Author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
* ProjectDate: 01/10/2014
* Version: Python 2.7.8 + OpenCV 2.4.10

## Project Description
* Find Face and its form and make a CSV database
* This project using 'HAAR' Classify windows to detect objects (eyes, nose, face, mouth)
* Detected Face will cropped by 160 x 160 px image for Normalization

## File & Folder Description
* haarClassifier : Classified xml data that is open source
* testImage : test image repository
* FaceDetection.py : testing file this script examine image and find facial objects

## What can & What can not be found..
### Can
* Face Shape (p.14)
* Eyes (p.13)
* Noes (p.13)
* Mouth (p.13)

### Can not
* Eyebrows (p.13)
* Dimension in target's face (p.15)
* Facial color (additional)
* Age (additional)
* Gender (additional, http://docs.opencv.org/trunk/modules/contrib/doc/facerec/tutorial/facerec_gender_classification.html)