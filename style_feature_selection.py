import numpy as np
from itertools import combinations, chain
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from character_recognition import *
from sklearn.model_selection import cross_val_score
import statistics

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
        #X_train, X_test, y_train, y_test = train_test_split(img_features, labels, test_size=0.25)
        svmClassifier = SVC()
        svm_score = statistics.mean(cross_val_score(svmClassifier, img_features, labels, cv=5))
        #svmClassifier.fit(img_features, labels)
        #svm_score = svmClassifier.score(X_test, y_test)
        print('SVM score: ' + str(svm_score))
        # RFClassifier = RandomForestClassifier(n_estimators=100, random_state=0)
        # RFClassifier.fit(X_train, y_train)
        # rf_score = RFClassifier.score(X_test, y_test)
        # print('Random Forest score: ' + str(rf_score))
        
        #if rf_score > rf_best:
        #   rf_best = rf_score
        if svm_score > svm_best:
            svm_best = svm_score
    return svm_best

image_features = np.load('style_augmented_feature_vecs.npy')
print(len(image_features))
labels = np.load('style_augmented_labels.npy')
print(len(labels))
svm_best = select_features(image_features, labels, features, 0, 0)
# print('SVC best: {}'.format(svm_best))
# print('Random Forest best: {}'.format(rf_best))

#X_train, X_test, y_train, y_test = train_test_split(image_features, labels, test_size=0.25)
# parameters = {
#     'n_estimators':[50, 100, 150],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 10, 50, 75, 100],
#     'min_samples_split': [2,3,5,8],
#     'max_samples': [0.5, 0.9],
#     'min_samples_leaf': [1,2,3],
#     'max_leaf_nodes': [None, 2, 5, 10]
# }
# RFClassifier = RandomForestClassifier()
# random_grid = GridSearchCV(RFClassifier, parameters, cv = 5, verbose=0, n_jobs=-1)
# random_grid.fit(image_features, labels)
# # score = random_grid.score(image_features, labels)
# # print(score)

# print(random_grid.best_score_)
# print(random_grid.cv_results_['params'][random_grid.best_index_])
#print(random_grid.cv_results_)


#################### MLP ####################
# parameters = {
#     'alpha': [ 0.01, 0.001, 0.0001],
#     'activation': ['logistic', 'relu'],
#     'solver': ['adam'],
#     'hidden_layer_sizes': [(50,50,50), (50,50), (100,), (100,100)]
# }
# for image in image_features:
#     image = image[71:149]
    
# for label in labels:
#     label = label[71:149]

# mlp = MLPClassifier(max_iter=1000, early_stopping=True)
# random_grid = GridSearchCV(mlp, parameters, cv = 5, verbose=1, n_jobs=-1)
# random_grid.fit(image_features, labels)
# score = random_grid.score(X_test, y_test)
# print(score)
# print(random_grid.best_score_)
# print(random_grid.cv_results_['params'][random_grid.best_index_])

#### SVC ####
# SVM score: 0.5753968253968254

# All features:
# (0, 1, 2, 3, 4, 5, 6, 7, 8)
# SVM score: 0.5317460317460317
################

#### RandomForest (100 estimators) unnormalized ####
# Best:
# (0, 1, 2, 3, 5, 6)
# Random Forest score: 0.8862433862433863

# All features:
# (0, 1, 2, 3, 4, 5, 6, 7, 8)
# Random Forest score: 0.876984126984127
################