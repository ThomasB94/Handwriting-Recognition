import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import math
from character_recognition import *
import random
import imutils

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
            if ((min_array[idx] != 0) and (max_array[idx] != 0)):
                image[idx] = (image[idx] - min_array[idx]) / (max_array[idx] - min_array[idx])
                #print(min_array[idx], max_array[idx])

def augment(image):
    if random.random() > 0.5:
        #rotate the image
        image = cv2.bitwise_not(image)
        angle = int(random.uniform(-30, 30))
        rotated = imutils.rotate_bound(image, angle)
        image = cv2.bitwise_not(image)
        return image
    else:
        #crop the image by 10
        crop = [0, 0, 0, 5]
        random.shuffle(crop)
        cropped = image[crop[0]:image.shape[0]-crop[1], crop[2]:image.shape[1]-crop[3]]
        return cropped

features = [[feature1, 0, 99999, 0, 10],
            [feature2, 0, 99999, 10, 20],
            [feature3, 0, 99999, 20, 28],
            [feature4, 0, 99999, 28, 36],
            [feature5, 0, 99999, 36, 56],
            [feature6, 0, 99999, 56, 60],
            [feature7, 0, 99999, 60, 67],
            [feature8, 0, 99999, 67, 71],
            [feature9, 0, 99999, 71, 149]
            ]

images = []
labels = []
print('reading images')

for dir in os.scandir("./characters_for_style_classification"):
    counter = 600
    for root, dirs, files in os.walk(dir.path, topdown=False):
        for name in files:
            # append the labels
            labels.append(dir.name)
                # Load image
            image = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)
            _, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            images.append(featureVect(image, features))
            counter = counter - 1
    while counter > 0:
        #get random image from directory
        fileList = [os.path.join(root,f) for root,dirs,files in os.walk(dir.path) for f in files]
        random_file=random.choice(fileList)
        labels.append(dir.name)
        #load image
        image = cv2.imread(random_file, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        #augment the image
        image = augment(image)
        #extract features
        images.append(featureVect(image, features))
        counter = counter - 1
            #print(dir)
'''
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
'''

print('normalizing')
normalize_images2(images, features)
print('saving images')
np.save('style_augmented_feature_vecs.npy', images)
np.save('style_augmented_labels.npy', labels)
print('extracted features')
#np.save('labels.npy', labels)
