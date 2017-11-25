''' Module containing methods for classification and evaluation of classifiers. '''
import os

from category_encoders.ordinal import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC

import io_util as io

class Classifier(object):
    ''' Class for classification. '''

    def __init__(self, dataset, path):
        self.dataset = dataset
        
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None
        
        self.accuracy_knn = 0
        self.accuracy_nb = 0
        self.accuracy_dt = 0
        self.accuracy_svm = 0
        self.accuracy_nc = 0

        self.encode_and_split(path)

    #toString() equivalent to Java.
    def __str__(self):
        return ("Classification Results:"
               "\nAccuracy NB:  " + '{0:.2f}%'.format(self.accuracy_nb) +
               "\nAccuracy kNN: " + '{0:.2f}%'.format(self.accuracy_knn) +
               "\nAccuracy DT:  " + '{0:.2f}%'.format(self.accuracy_dt) +
               "\nAccuracy SVM: " + '{0:.2f}%'.format(self.accuracy_svm) +
               "\nAccuracy NC:  " + '{0:.2f}%'.format(self.accuracy_nc) +
               "\n\nMax.: " + '{0:.2f}%'.format(max(self.accuracy_nb, self.accuracy_knn, self.accuracy_dt, self.accuracy_svm, self.accuracy_nc)))

    def encode_and_split(self, path):
        ''' Encode datatset and split it into training and test data. '''
        if os.path.exists(path):
            data_encoded = io.read_csv(path)
        else:
            encoder = OrdinalEncoder()
            data_encoded = encoder.fit_transform(self.dataset)
            print('Encoding done for file: ' + str(path))
            io.write_csv(data_encoded, path)

        target = data_encoded['perceived_quality']
        data_encoded.drop('perceived_quality', axis=1, inplace=True)

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(data_encoded, target, test_size=0.2, random_state=42, stratify=target)

    def classify_nb(self): 
        ''' Classification with Naive Bayes. '''
        naive_bayes = GaussianNB()
        naive_bayes.fit(self.data_train, self.target_train)
        prediction = naive_bayes.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_nb:
            self.accuracy_nb = acc

    def classify_knn(self, n=5):
        ''' Classification with K_Nearest_Neighbor. '''
        knn_estimator = KNeighborsClassifier(n)
        knn_estimator.fit(self.data_train, self.target_train)
        prediction = knn_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_knn:
            self.accuracy_knn = acc

    def classify_nc(self):
        ''' Classification with Nearest Centroid. '''
        nc_estimator = NearestCentroid()
        nc_estimator.fit(self.data_train, self.target_train)
        prediction = nc_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_nc:
            self.accuracy_nc = acc

    def classify_svm(self): #, C=1.0, gamma ='auto')
        ''' Classification with Support Vector Machine. '''
        svm_estimator = SVC()
        svm_estimator.fit(self.data_train, self.target_train)
        prediction = svm_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_svm:
            self.accuracy_svm = acc

    def classify_dt(self):
        ''' Clasificiation with Decision Tree'''
        decision_tree = tree.DecisionTreeClassifier(max_depth=3,criterion="gini")
        decision_tree.fit(self.data_train,self.target_train) 
        prediction = decision_tree.predict(self.data_test)
        self.accuracy_dt = decision_tree.score(self.target_test,prediction)
        acc = accuracy_score(self.target_test,prediction) * 100
        if acc > self.accuracy_dt:
            self.accuracy_dt = acc