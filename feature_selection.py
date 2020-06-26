import numpy as np
from itertools import combinations, chain
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from character_recognition import *

features = [[feature1, 0, 99999, 0, 10],
            [feature2, 0, 99999, 10, 20],
            [feature3, 0, 99999, 20, 28],
            [feature4, 0, 99999, 28, 36],
            [feature5, 0, 99999, 36, 56],
            [feature6, 0, 99999, 56, 60],
            [feature7, 0, 99999, 60, 67],
            [feature8, 0, 99999, 67, 71],
            ]

def select_features(image_features, labels, features, svm_best, rf_best):
    # Create all possible subsets of possible feature combinations
    allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))
    subsets = allsubsets(len(features))
    subsets = subsets[1:]
    for subset in subsets:
        print(subset)
        indices = list(subset)
        img_features = []
        for image in image_features:
            single_img_features = []
            single_img_features = np.array(single_img_features)
            for index in indices:
                single_img_features = np.append(single_img_features, image[features[index][3]:features[index][4]])
            img_features.append(single_img_features)
        X_train, X_test, y_train, y_test = train_test_split(img_features, labels, test_size=0.25, random_state=42)
        #svmClassifier = svm.SVC()
        #svmClassifier = svm.LinearSVC()
        #svmClassifier = svm.NuSVC()
        #svmClassifier.fit(X_train, y_train)
        #svm_score = svmClassifier.score(X_test, y_test)
        #print('SVM score: ' + str(svm_score))
        RFClassifier = RandomForestClassifier(n_estimators=100, random_state=0)
        RFClassifier.fit(X_train, y_train)
        rf_score = RFClassifier.score(X_test, y_test)
        print('Random Forest score: ' + str(rf_score))
        
        if rf_score > rf_best:
           rf_best = rf_score
        # if svm_score > svm_best:
        #     svm_best = svm_score
    return rf_best

image_features = np.load('numpy_arrays\\300_augmented_feature_vecs.npy')
print(len(image_features))
labels = np.load('numpy_arrays\\300_augmented_labels.npy')
print(len(labels))
#best = select_features(image_features, labels, features, 0, 0)
#print(best)

# X_train, X_test, y_train, y_test = train_test_split(image_features, labels, test_size=0.25, random_state=42)
parameters = {
    'n_estimators':[50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 50, 75, 100],
    'min_samples_split': [2,3,5,8],
    'max_samples': [0.5, 0.9],
    'min_samples_leaf': [1,2,3],
    'max_leaf_nodes': [None, 2, 5, 10]
}
RFClassifier = RandomForestClassifier()
random_grid = GridSearchCV(RFClassifier, parameters, cv = 5, n_jobs=2)
random_grid.fit(image_features, labels)
#score = random_grid.score(X_test, y_test)
#print(score)

print(random_grid.best_score_)
print(random_grid.cv_results_['params'][random_grid.best_index_])
#print(random_grid.cv_results_)

#### SVC ####
# (0, 1, 3, 4, 5)
# (0, 1, 3, 5, 7)
# (0, 2, 3, 4, 5)
# (0, 1, 2, 3, 5, 7)
# SVM score: 0.8597530864197531

# All features:
# (0, 1, 2, 3, 4, 5, 6, 7)
# SVM score: 0.8330864197530864
################

#### RandomForest (100 estimators) unnormalized ####
# Best:
# (0, 2, 3, 4, 5, 7)
# Random Forest score: 0.9244444444444444

# All features:
# (0, 1, 2, 3, 4, 5, 6, 7)
# Random Forest score: 0.92
################