import pandas as pd
import io_util as io
import numpy as np
import collections

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
#import graphviz
#import pydotplus
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.multiclass import unique_labels
from category_encoders.ordinal import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid



def naive_bayes(dataset):
    '''Trains and tests a Naive Bayes Classifier with selected features'''
    ###können wir das nicht für alle methoden verwenden bis zu data_train, data_test,...?
    encoder = OrdinalEncoder()
    dataset_encoded = encoder.fit_transform(dataset)
    listings_data=dataset_encoded.drop(columns=['id','perceived_quality'])
    listings_target= dataset_encoded['perceived_quality']
    
    data_train, data_test, target_train, target_test = io.split_dataset_regular(listings_data,listings_target)
    
    naive_bayes = GaussianNB()
    naive_bayes.fit(data_train, target_train)
    prediction = naive_bayes.predict(data_test)

    #nbresults=pd.DataFrame(data_test)
    #quality_predicted = nbresults.assign(predicted_quality=prediction)
    #io.write_csv(quality_predicted, '../data/playground/naivebayes.csv')
    
    accuracy=accuracy_score(target_test, prediction)
    print('Accuracy of Naive Bayes Classifier:{}'.format(accuracy))



# decisiontree: works with encoded data (use of Nadja's methood, encoded set NOT pushed, decisiontree saved as png file in playground
# BEFORE executing, we have to talk about the encode
def dt(dataset):
    '''DT'''
    encoder = OrdinalEncoder()
    dataset_encoded = encoder.fit_transform(dataset)
    listings_data=dataset_encoded.drop(columns=['id','perceived_quality'])
    listings_target= dataset_encoded['perceived_quality']

    data_train, data_test, target_train, target_test = io.split_dataset_regular(listings_data,listings_target)
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(data_train,target_train)
    prediction = decision_tree.predict(data_test)
    accuracy = accuracy_score(target_test,prediction)
    print('Accuracy of DT Classifier{}'.format(accuracy))



def knn(dataset):
    '''KNN'''
    encoder = OrdinalEncoder()
    dataset_encoded = encoder.fit_transform(dataset)
    listings_data=dataset_encoded.drop(columns=['id','perceived_quality'])
    listings_target= dataset_encoded['perceived_quality']

    data_train, data_test, target_train, target_test = io.split_dataset_regular(listings_data,listings_target)

    knn_estimator = KNeighborsClassifier(4)
    knn_estimator.fit(data_train, target_train)

    prediction = knn_estimator.predict(data_test)
    #print('Prediction of KNN Classifier:{}'.format(predict))

    accuracy = accuracy_score(target_test, prediction)
    print('Accuracy of KNN Classifier:{}'.format(accuracy))

def nearest_centroid(dataset):
    encoder = OrdinalEncoder()
    dataset_encoded = encoder.fit_transform(dataset)
    listings_data=dataset_encoded.drop(columns=['id','perceived_quality'])
    listings_target= dataset_encoded['perceived_quality']

    data_train, data_test, target_train, target_test = io.split_dataset_regular(listings_data,listings_target)
    
    nearest_cent = NearestCentroid()
    nearest_cent.fit(data_train, target_train)
    prediction = nearest_cent.predict(data_test)

    accuracy = accuracy_score(target_test, prediction)
    print('Accuracy of Nearest Centroid Classifier:{}'.format(accuracy))


'''Run classifiers'''
dataset = io.read_csv('../data/playground/dataset.csv')
#naive_bayes(dataset)
#knn(dataset)
#dt(dataset)
nearest_centroid(dataset)


