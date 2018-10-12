#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:03:02 2018

@author: Saicharan
"""

from imutils import face_utils
import imutils
import numpy as np
import argparse
import dlib
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-p","--shape-predictor",required=True,help="path to face detector")
ap.add_argument("-i","--image",required=True,help="path to image")
args = vars(ap.parse_args())

#facial detector and landmark detector (HOG based)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grayscaling
image = cv2.imread(args['image'])
image = imutils.resize(image,width=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Detect facess
rects = detector(gray,1)

for (i,rect) in enumerate(rects):
    
    shape = predictor(gray, rect)# shape object contains all 68 facial landmarks
    shape = face_utils.shape_to_np(shape) #converting into np array
    
    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
	# show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
    for (x, y) in shape:
	    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
    


