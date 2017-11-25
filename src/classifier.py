''' Module containing methods for classification and evaluation of classifiers. '''
import pandas as pd
import io_util as io

from category_encoders.ordinal import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC

class Classifier(object):
    ''' Class for classification. '''

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_encoded = None
        
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None
        
        self.accuracy_knn = 0
        self.accuracy_nb = 0
        self.accuracy_dt = 0
        self.accuracy_svm = 0
        self.accuracy_nc = 0

        self.encode_and_split()

    #toString() equivalent to Java.
    def __str__(self):
        return ("Classification Results:"
               "\nAccuracy NB:  " + '{0:.2f}%'.format(self.accuracy_nb) +
               "\nAccuracy kNN: " + '{0:.2f}%'.format(self.accuracy_knn) +
               "\nAccuracy DT:  " + '{0:.2f}%'.format(self.accuracy_dt) +
               "\nAccuracy SVM: " + '{0:.2f}%'.format(self.accuracy_svm) +
               "\nAccuracy NC:  " + '{0:.2f}%'.format(self.accuracy_nc) +
               "\n\nMax.: " + str(max(self.accuracy_nb, self.accuracy_knn, self.accuracy_dt, self.accuracy_svm, self.accuracy_nc)))

    def encode_and_split(self):
        ''' Encode datatset and split it into training and test data. '''
        encoder = OrdinalEncoder()
        self.dataset_encoded = encoder.fit_transform(self.dataset[['experiences_offered', 'host_location', 'host_response_time', 'host_is_superhost', 'neighbourhood_cleansed', 'instant_bookable', 'cancellation_policy',
                                                                   'require_guest_profile_picture', 'require_guest_phone_verification', 'calculated_host_listings_count', 'reviews_per_month', 'host_response_rate_binned', 'host_verification_binned']])
        
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(self.dataset_encoded, self.dataset['perceived_quality'], test_size=0.2, random_state=42, stratify=self.dataset['perceived_quality'])

    def classify_nb(self): 
        ''' Classification with Naive Bayes. '''
        naive_bayes = GaussianNB()
        naive_bayes.fit(self.data_train, self.target_train)
        prediction = naive_bayes.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction)
        if acc > self.accuracy_nb:
            self.accuracy_nb = acc

    def classify_knn(self, n=5):
        ''' Classification with K_Nearest_Neighbor. '''
        knn_estimator = KNeighborsClassifier(n)
        knn_estimator.fit(self.data_train, self.target_train)
        prediction = knn_estimator.predict(self.data_test)
        acc = knn_estimator.score(self.target_test, prediction)
        if acc > self.accuracy_knn:
            self.accuracy_knn = acc

    def classify_nc(self):
        ''' Classification with Nearest Centroid. '''
        nc_estimator = NearestCentroid()
        nc_estimator.fit(self.data_train, self.target_train)
        prediction = nc_estimator.predict(self.data_test)
        acc = nc_estimator.score(self.target_test, prediction)
        if acc > self.accuracy_nc:
            self.accuracy_nc = acc

    def classify_svm(self, c, gamma): #, C=1.0, gamma ='auto')
        ''' Classification with Support Vector Machine. '''
        svc = SVC(c=c, gamma=gamma)
        svc.fit(self.data_train, self.target_train)
        prediction = svc.predict(self.data_test)
        acc = svc.score(self.target_test, prediction)
        if acc > self.accuracy_svm:
            self.accuracy_svm = acc