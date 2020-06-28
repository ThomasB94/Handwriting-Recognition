import pickle
import numpy as np
import math
import cv2
import os

from .character_recognition import *

class Recognizer:
    def __init__(self):
        with open('recognition/character_recognizer.pickle', 'rb') as pfile:
           self.char_model = pickle.load(pfile)
        with open('recognition/style_classifier.pickle', 'rb') as pfile:
           self.style_model = pickle.load(pfile)
        self.style_vect = [0, 0, 0]
        self.styles = ["Archaic", "Hasmonean", "Herodian"]
        self.feature_list = [
                    [feature1],
                    [feature2],
                    [feature3],
                    [feature4],
                    [feature5],
                    [feature6],
                    [feature7],
                    [feature8],
                    ]

    def _create_feature_vect(self, image):
        #uses function from character_recognition
        return featureVect(image, self.feature_list)

    def _predict_character(self, image_features_vector):
        return self.char_model.predict(image_features_vector)

    def _probability_check(self, image_features_vector):
        probs = self.char_model.predict_proba(image_features_vector)
        print(probs)

    def _predict_style(self, image_features_vector):
        return self.style_model.predict(image_features_vector)

    def _record_style(self, style):
        #records the style in the class
        self.style_vect[self.styles.index(style)] = self.style_vect[self.styles.index(style)] + 1

    def get_style(self):
        prediction = self.styles[np.argmax(self.style_vect)] 
        self._page_reset()
        return prediction

    def _page_reset(self):
        self.style_vect = [0, 0, 0]

    def predict(self, image):
        image_features = self._create_feature_vect(image).reshape(1, -1)
        char = self._predict_character(image_features)[0]
        style = self._predict_style(image_features)[0]
        self._record_style(style)
        return char

# predictor = Recognizer()
# image = cv2.imread("./characters_for_style_classification/Hasmonean/Bet/Bet_00.jpg", cv2.IMREAD_GRAYSCALE)
# _, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
# print(image.shape)
# print(predictor.predict(image))
# print(predictor.get_style())
