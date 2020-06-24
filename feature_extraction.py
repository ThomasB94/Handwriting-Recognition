import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import math
from character_recognition import *

def normalize_images(images, features):
    for image in images:
        for feature in features:
            if max(image[feature[3]:feature[4]]) > feature[1]:
                feature[1] = max(image[feature[3]:feature[4]])
            if min(image[feature[3]:feature[4]]) < feature[2]:
                feature[2] = min(image[feature[3]:feature[4]])
    
    for image in images:
        for feature in features:
            for idx in range(feature[3], feature[4]):
                image[idx] = (image[idx] - feature[2]) / (feature[1] - feature[2])

def normalize_images2(images, features):
    length = len(images[0])
    max_array = [0 for _ in range(0, length)]
    min_array = [99999 for _ in range(0, length)]
    for image in images:
        for idx in range(0, length):
            if image[idx] > max_array[idx]:
                max_array[idx] = image[idx]
            if image[idx] < min_array[idx]:
                min_array[idx] = image[idx]
    for image in images:
        for idx in range(0, length):
            image[idx] = (image[idx] - min_array[idx]) / (max_array[idx] - min_array[idx])   

features = [[feature1, 0, 99999, 0, 10],
            [feature2, 0, 99999, 10, 20],
            [feature3, 0, 99999, 20, 28],
            [feature4, 0, 99999, 28, 36],
            [feature5, 0, 99999, 36, 56],
            [feature6, 0, 99999, 56, 60],
            [feature7, 0, 99999, 60, 67],
            [feature8, 0, 99999, 67, 71],
            ]

images = []
labels = []
print('reading images')
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if name.endswith(".jpg"):
            # Extract the label from the path
            label = root.split('\\', 3)
            labels.append(label[2])
            # Load image 
            image = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)
            _, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            images.append(featureVect(image, features))

print('normalizing')
normalize_images2(images, features)
print('saving images')
np.save('normalized_feature_vecs.npy', images)
print('extracted features')
#np.save('labels.npy', labels)
