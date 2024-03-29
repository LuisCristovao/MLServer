# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:27:49 2018

@author: lcristovao

Objective of this program is to make a function that does all the machine learning Work
I mean it returns the best predictor ready to work. It must be capable to work on multiclass 
datasets. The Values of data set must be numeric or categorical (string)
"""

from calendar import c
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import joblib

#from sklearn.externals import joblib


class Predictor:

    loading = str(0)+'%'
    score = 0
    best_model_str = ""

    @staticmethod
    def categoricalToNumeric(array):
        le = preprocessing.LabelEncoder()
        le.fit(array)
        return le.transform(array)

    def TurnDatasetToNumeric(self, dataset):

        for i in range(len(dataset.dtypes)):
            if dataset.dtypes[i] == object:
                v = dataset.iloc[:, i].values
                # print(v)
                v = self.categoricalToNumeric(v)
                dataset.iloc[:, i] = v

        return dataset

    def ReturnPredictor(self, dataset, true_test_size=0.6, validation_size=0.20, cross_val_splits=10, seed=7, SVM_data_size=1000):
        '''
        dataset must have this format: Atribute1|Atribute2|...|Class|
        the atributes must be numerical! the class doesn't
        '''
        class_index = dataset.shape[1]-1
        # Shuffle data
        dataset = dataset.sample(frac=1, random_state=seed)
        # print(dataset.head)
        self.loading = str(5)+'%'

        # Turn all columns of atributes that have string categorical values to numbers
        dataset.iloc[:, :-1] = self.TurnDatasetToNumeric(dataset.iloc[:, :-1])
        self.loading = str(10)+'%'

        # Train Validation Part-----------------------------------------------------
        array = dataset.values
        X = array[:, 0:class_index]
        Y = array[:, class_index]
        # print(Y)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
            X, Y, test_size=validation_size, random_state=seed)
        self.loading = str(15)+'%'
        scoring = 'accuracy'

        # Spot Check Algorithms
        models = []
        models.append(('LR', LogisticRegression()))
        # self.loading=str(20)+'%'
        models.append(('LDA', LinearDiscriminantAnalysis()))
        # self.loading=str(30)+'%'
        models.append(('KNN-default', KNeighborsClassifier()))
        models.append(('KNN-1', KNeighborsClassifier(n_neighbors=1)))
        # self.loading=str(40)+'%'
        models.append(('CART', DecisionTreeClassifier()))
        # self.loading=str(50)+'%'
        models.append(('NB', GaussianNB()))
        # self.loading=str(70)+'%'
        if(dataset.shape[0] < SVM_data_size):
            models.append(('SVM', svm.SVC()))
        # evaluate each model in turn
        self.loading = str(20)+'%'
        mean_results = []
        model_index = 0
        prev_loading = 20

        is_small_data = dataset.shape[0] < 30

        if is_small_data:
            models = list(
                filter(lambda model: model[0] != "KNN-default", models))

        for name, model in models:
            #kfold = model_selection.KFold(n_splits=cross_val_splits, random_state=seed)
            if is_small_data:
                model.fit(dataset.iloc[:,:class_index],dataset.iloc[:,class_index])
                _predictions = model.predict(dataset.iloc[:, :class_index])
                _score = accuracy_score(
                    dataset.iloc[:, class_index], _predictions)
                mean_results.append(_score)
                msg = "%s: %f " % (
                    name, _score )
                print(msg)
            else:
                model.fit(X_train, Y_train)
                cv_results = model_selection.cross_val_score(
                    model, X_validation, Y_validation)
                mean_results.append(cv_results.mean())
                msg = "%s: %f (%f)" % (
                    name, cv_results.mean(), cv_results.std())
                print(msg)

            model_index += 1
            self.loading = str(prev_loading + 10*model_index)+'%'

        self.loading = str(90)+'%'
        mean_results = np.array(mean_results)
        best_model_index = mean_results.argmax()
        best_model = models[best_model_index][1]
        print("Best Model: ", models[best_model_index][0])
        self.best_model_str = ""
        self.best_model_str += models[best_model_index][0]

        # Fit best model In Training_validation dataset

        self.loading = str(95)+'%'
        FFArray = dataset.values
        FFX = FFArray[:, 0:class_index]
        FFY = FFArray[:, class_index]
        predictions = best_model.predict(FFX)
        self.score = accuracy_score(FFY, predictions)
        print('\n\nFinal Final Score: ', self.score)
        print('Confusion matrix\n', confusion_matrix(FFY, predictions))
        self.loading = str(100)+'%'
        return best_model

    @staticmethod
    def ExportModel(model, path_and_name):
        print("Exporting Model")
        joblib.dump(model, path_and_name)


# _______________Main____________________________________________________________
print(__name__)
# Load dataset from site
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pd.read_csv(url, names=names)
#
# best_model=ReturnPredictor(dataset)
#
# values=dataset.values
# np.random.shuffle(values)
# Number of tests
# n=10
#print("Original values:\n",values[:n,:])
# for i in range(n):
#    print("Prediction for previous line:\n",values[i,:-1],"->",best_model.predict(values[:n,:-1])[i])
#
