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
* logicLibrary.py: like a header file, classifiers and functions were declared

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

## REFERENCES
* OpenCV reference: http://goo.gl/Ak5z1u
* Haar Cascades: http://alereimondo.no-ip.org/OpenCV/34
* Referred Blog: https://realpython.com/blog/python/face-recognition-with-python/
* Crop Face y:y+h, x:x+w (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python#_=_)
* Resize: http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html