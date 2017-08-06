#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:28:54 2017

@author: johnkenny
"""

import os
import time
import cv2
import numpy as np
from PIL import Image

#pip install Pillow
#conda install -c menpo opencv3=3.1.0

'''
Used to get all image files
'''
def get_training_data(face_cascade, data_dir):
    images = []
    labels = []
    #used to get all the images except files that end in .wink as these images will be used to test the as
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith('.wink')]

    #loop through images and assign images and lables
    for image_file in image_files:
        #load image and convert to grayscale
        img = Image.open(image_file).convert('L')
        img = np.array(img, np.uint8)
        filename = os.path.split(image_file)[1]
        #compare ai predicted 
        true_person_number = int(filename.split('.')[0].replace('subject', ''))
        
        #detect face
        faces = face_cascade.detectMultiScale(img, 1.05, 6)
        
        #cut each face out of the image
        for face in faces:
            x, y, w, h = face
            faceReg = img[y:y+h, x:x+w]
            faceReg = cv2.resize(faceReg, (150,150))
            #add face to images array
            images.append(faceReg)
            #add the image id
            labels.append(true_person_number)

    return images, labels

#compare image --used for testing ai
def evaluate(recognizer, face_cascade, data_dir):
    #only get the wink images
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wink')]
    
    #matches
    correct = 0

    for image_file in image_files:
        img = Image.open(image_file).convert('L')
        img = np.array(img, np.uint8)

        filename = os.path.split(image_file)[1]
        #compare ai predicted 
        true_person_number = int(filename.split('.')[0].replace('subject', ''))
        
        
        
        #detect face
        faces = face_cascade.detectMultiScale(img, 1.05, 6)
        
        #cut each face out of the image
        for face in faces:
            x, y, w, h = face
            faceReg = img[y:y+h, x:x+w]
            faceReg = cv2.resize(faceReg, (150,150))
            
            predictedId  = recognizer.predict(faceReg)
            
            
            #check id its correct
            if predictedId == true_person_number:
                correct = correct + 1
                print ("matched correct %d" % predictedId)
            else:
                print("incorrect %d as %f" % (true_person_number, predictedId))
                
                
    #how accurate was Joe
    accurate = correct / float(len(image_files)) * 100
    
    print("matched correct %d%%" % accurate)
        
#can be used with new iamges      
def predict(recognizer, face_cascade, img):
    predictions = []

    
    #detect face
    faces = face_cascade.detectMultiScale(img, 1.05, 6)
        
    #cut each face out of the image
    for face in faces:
        x, y, w, h = face
        faceReg = img[y:y+h, x:x+w]
        faceReg = cv2.resize(faceReg, (150,150))
        
        predictions.append(recognizer.predict(faceReg))
    
    return predictions
        
        
#set up ai
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognition = cv2.face.createLBPHFaceRecognizer()
#face_recognition = cv2.face.createEigenFaceRecognizer()
#face_recognition = cv2.face.createFisherFaceRecognizer()

print("reading sample data")
start = time.time()
images, labels = get_training_data(face_cascade, "yalefaces")
face_recognition.train(images, np.array(labels))
print("sample data complete")
print(time.time() - start)
#test ai
evaluate(face_recognition, face_cascade, "yalefaces")
print("new")
img = cv2.imread("elvis.jpg", cv2.IMREAD_GRAYSCALE)
print (predict(face_recognition, face_cascade, img))

video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
	#frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
    faces = face_cascade.detectMultiScale(gray, 1.05, 6)
    for face in faces:
        x, y, w, h = face
        face_region = gray[y:y+h, x:x+w]
		
        dicted_person_number = face_recognition.predict(face_region)
		
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        #add your image to the image database
        if dicted_person_number == 2:
            cv2.putText(frame, "username here", (x,y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
        else:
            cv2.putText(frame, str(dicted_person_number), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
			
    cv2.imshow('Running face recognition...', frame)
	
    if cv2.waitKey(1) & 0xFF == ord('q'): break
	
video_cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)   
    
    