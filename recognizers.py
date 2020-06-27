import pickle
import numpy as np
import math
import cv2
import os

from character_recognition import *



class recognizer:

    def __init__(self):
        with open('character_recognizer.pickle', 'rb') as pfile:
            self.char_model = pickle.load(pfile)
        with open('style_classifier.pickle', 'rb') as pfile:
            self.style_model = pickle.load(pfile)
        self.styleVect = [0, 0, 0]
        self.styles = ["Archaic", "Hasmonean", "Herodian"]
        self.featureList = [
                    [feature1],
                    [feature2],
                    [feature3],
                    [feature4],
                    [feature5],
                    [feature6],
                    [feature7],
                    [feature8],
                    ]

    def createFeatureVect(self, image):
        #uses function from character_recognition
        return featureVect(image, self.featureList)

    def predictCharacter(self, image_features_vector):
        return self.char_model.predict(image_features_vector)

    def probabilityCheck(self, image_features_vector):
        probs = self.char_model.predict_proba(image_features_vector)
        print(probs)

    def predictStyle(self, image_features_vector):
        return self.style_model.predict(image_features_vector)

    def recordStyle(self, style):
        #records the style in the class
        self.styleVect[self.styles.index(style)] = self.styleVect[self.styles.index(style)] + 1

    def getStyle(self):
        return self.styles[np.argmax(self.styleVect)]
        self.pageReset()

    def pageReset(self):
        self.styleVect = [0, 0, 0]

    def predict(self, image):
        image_features = self.createFeatureVect(image).reshape(1, -1)
        char = self.predictCharacter(image_features)[0]
        style = self.predictStyle(image_features)[0]
        self.recordStyle(style)
        return char

predictor = recognizer()
image = cv2.imread("./characters_for_style_classification/Archaic/Alef/Alef_00.jpg", cv2.IMREAD_GRAYSCALE)
_, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
print(image.shape)
predictor.predict(image)
print(predictor.getStyle())
