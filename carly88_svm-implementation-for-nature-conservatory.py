#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
import cv2
import csv
import os
from os import listdir
from os.path import isfile, join

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
mydirvalues = [d for d in os.listdir(os.path.dirname(os.path.abspath(__file__)))]
print(mydirvalues)
onlyfiles = [f for f in listdir("../input/train/") if isfile(join("../input/train/", f))]
print(onlyfiles)

dir_names = [d for d in listdir("../input/train/") if not isfile(join("../input/train/", d))]
print(dir_names)

file_paths = {}
class_num = 0
for d in dir_names:
     fnames = [f for f in listdir("../input/train/"+d+"/") if isfile(join("../input/train/"+d+"/", f))]
     print(fnames)
     file_paths[(d, class_num, "../input/train/"+d+"/")] = fnames
     class_num += 1




# General steps:
# Extract feature from each file as HOG or similar... or SIFT... or Similar...
# map each to feature space... and train some kind of classifier on that. SVM is a good choice.
# do the same for each feature in test set...
training_data = np.array([])
training_labels = np.array([])

for key in file_paths:
    category = key[1]
    directory_path = key[2]
    file_list = file_paths[key]

    # shuffle this list, so we get random examples
    np.random.shuffle(file_list)
    
    # Stop early, while testing, so it doesn't take FOR-EV-ER (FOR-EV-ER)
    i = 0

    # read in the file and get its SIFT features
    for fname in file_list:
        fpath = directory_path + fname
        print(fpath)
        print("Category = " + str(category))
        # extract features!
        gray = cv2.imread(fpath,0)
        gray = cv2.resize(gray, (400, 250))  # resize so we're always comparing same-sized images
                                             # Could also make images larger/smaller
                                             # to tune for greater accuracy / more speedd
                    
        """ My Choice: SIFT (Scale Invariant Feature Transform)"""
        # However, this does not work on the Kaggle server
        # because it's in a separate package in the opencv version used on the Kaggle server.
        # This is a very robust method however, worth trying when it's reasonable to do so. 
        detector = cv2.SIFT()
        kp1, des1 = detector.detectAndCompute(gray, None)
        
        """ Another option that will work on Kaggle server is ORB"""
        # find the keypoints with ORB
        #kp = cv2.orb.detect(img,None)
        # compute the descriptors with ORB
        #kp1, des1 = cv2.orb.compute(img, kp)
        
        """ Histogram of Gradients - often used to for detected people/animals in photos"""
         # Havent' tried this one in the SVM yet, but here's how to get the HoG, using openCV
         # hog = cv2.HOGDescriptor()
         #img = cv2.imread(sample)
         # h = hog.compute(im)

        # This is to make sure we have at least 100 keypoints to analyze
        # could also duplicate a few features if needed to hit a higher value
        if len(kp1) < 100:
            continue
            
        # transform the data to float and shuffle all keypoints
        # so we get a random sampling from each image
        des1 = des1.astype(np.float64)
        np.random.shuffle(des1)
        des1 = des1[0:100,:] # trim vector so all are same size
        vector_data = des1.reshape(1,12800) 
        list_data = vector_data.tolist()

        # We need to concatenate ont the full list of features extracted from each image
        if len(training_data) == 0:
            training_data = np.append(training_data, vector_data)
            training_data = training_data.reshape(1,12800)
        else:
            training_data   = np.concatenate((training_data, vector_data), axis=0)
            
        training_labels = np.append(training_labels,category)

        # early stop
        i += 1
        if i > 50:
            break




# Alright! Now we've got features extracted and labels
X = training_data
y = training_labels
y = y.reshape(y.shape[0],)

# Create and fit the SVM
# Fitting should take a few minutes
clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(X,y)




# Now, extract one of the images and predict it
gray = cv2.imread('../inputtest_stg1/img_00071.jpg', 0)  # Correct is LAG --> Class 3
kp1, des1 = detector.detectAndCompute(gray, None)

des1 = des1[0:100, :]   # trim vector so all are same size
vector_data = des1.reshape(1, 12800)

print("Linear SVM Prediction:")
print(clf.predict(vector_data))        # prints highest probability class, only
print(clf.predict_proba(vector_data))  # shows all probabilities for each class. 
                                       #    need this for the competition




# save SVM model
# joblib.dump(clf, 'filename.pkl')
# to load SVM model, use:  clf = joblib.load('filename.pkl')




# early stoppage...
# only do 10
i = 0
for f in fnames:
    file_name = "test_stg1/" + f
    print("---Evaluating File at: " + file_name)
    gray = cv2.imread(file_name, 0)  # Correct is LAG --> Class 3
    gray = cv2.resize(gray, (400, 250))  # resize so we're always comparing same-sized images
    kp1, des1 = detector.detectAndCompute(gray, None)

    # ensure we have at least 100 keypoints to analyze
    if len(kp1) < 100:
        # and duplicate some points if necessary
        current_len = len(kp1)
        vectors_needed = 100 - current_len
        repeated_vectors = des1[0:vectors_needed, :]
        # concatenate repeats onto des1
        while len(des1) < 100:
            des1 = np.concatenate((des1, repeated_vectors), axis=0)
        # duplicate data just so we can run the model.
        des1[current_len:100, :] = des1[0:vectors_needed, :]

    np.random.shuffle(des1)  # shuffle the vector so we get a representative sample
    des1 = des1[0:100, :]   # trim vector so all are same size
    vector_data = des1.reshape(1, 12800)
    print("Linear SVM Prediction:")
    print(clf.predict(vector_data))
    svm_prediction = clf.predict_proba(vector_data)
    print(svm_prediction)
    
    # format list for csv output
    csv_output_list = []
    csv_output_list.append(f)
    for elem in svm_prediction:      
        for value in elem:
            csv_output_list.append(value)

    # append filename to make sure we have right format to write to csv
    print("CSV Output List Formatted:")
    print(csv_output_list)

    # and append this file to the output_list (of lists)
    prediction_output_list.append(csv_output_list)

    # Uncomment to stop early
    if i > 10:
        break
    i += 1




# Write to csv
print(prediction_output_list[0:5])
"""  Uncomment to write to your CSV. Can't do this on Kaggle server directly.
try:
    with open("sift_and_svm_submission.csv", "wb") as f:
        writer = csv.writer(f)
        headers = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        writer.writerow(headers)
        writer.writerows(prediction_output_list)
finally:
    f.close()
"""

