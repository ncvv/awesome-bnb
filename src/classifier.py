''' Module containing methods for classification and evaluation of classifiers. '''
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from category_encoders.ordinal import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels

import io_util as io

class Classifier(object):
    ''' Class for classification. '''

    def __init__(self, dataset, path, long_tfidf, display_columns):
        self.dataset = dataset
        self.data_encoded = None
        self.label = None
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None
        self.data_encoded_with_label =None
        self.accuracy_knn = 0
        self.accuracy_knn_n = "?"
        self.accuracy_nb = 0
        self.accuracy_dt = 0
        self.accuracy_svm = 0
        self.accuracy_nc = 0

        self.binary_labels = ["Bad", "Good"]
        self.long_tfidf = long_tfidf
        self.display_columns = display_columns
        self.roc_estimators = {}
        
        self.encode_and_split(path)

    #toString() equivalent to Java.
    def __str__(self):
        return ("Classification Results:"
               "\nAccuracy NB:  " + '{0:.2f}%'.format(self.accuracy_nb)  +
               "\nAccuracy kNN: " + '{0:.2f}%'.format(self.accuracy_knn) + ', n=' + str(self.accuracy_knn_n) +
               "\nAccuracy DT:  " + '{0:.2f}%'.format(self.accuracy_dt)  +
               "\nAccuracy SVM: " + '{0:.2f}%'.format(self.accuracy_svm) +
               "\nAccuracy NC:  " + '{0:.2f}%'.format(self.accuracy_nc)  +
               "\n\nMax.: " + '{0:.2f}%'.format(max(self.accuracy_nb, self.accuracy_knn, self.accuracy_dt, self.accuracy_svm, self.accuracy_nc)))

    def encode_and_split(self, path):
        ''' Encode datatset and split it into training and test data. '''
        if os.path.exists(path):
            self.data_encoded = io.read_csv(path)
        else:
            encoder = OrdinalEncoder()
            self.data_encoded = encoder.fit_transform(self.dataset)
            print('Encoding done for file: ' + str(path))
            io.write_csv(self.data_encoded, path)
       
        target = self.data_encoded['perceived_quality']
        self.label = target

        # Insert drop of single columns here
        drop_list = ['id', 'perceived_quality', 'instant_bookable',  'availability_30' ,'availability_60' ,'availability_90','availability_365', 'host_location', 'host_verification_binned','host_response_rate_binned','require_guest_phone_verification','host_response_time', 'zipcode','require_guest_profile_picture','calculated_host_listings_count', 'first_review', 'last_review']
        self.data_encoded.drop(drop_list, axis=1, inplace=True)

        # Insert drop of tfidf/amenity columns here
        self.exclude_amenity_columns()
        #self.exclude_transit_tfidf()
        #self.exclude_description_tfidf()
        #self.exclude_houserules_tfidf()
        self.exclude_neighbor_tfidf()
        self.data_encoded_with_label =  pd.concat([self.data_encoded, target], axis=1)
        if self.display_columns:
            print('Columns:\n' + '\n'.join(list(self.data_encoded)) + '\n')
    
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(self.data_encoded, target, test_size=0.2, random_state=42, stratify=target)

    def __drop(self, start, end):
        self.data_encoded.drop(self.data_encoded.columns[self.data_encoded.columns.get_loc(start): self.data_encoded.columns.get_loc(end) + 1], axis=1, inplace=True)

    def __print_cm(self, y_true, y_pred, labels, print_label):
        cm = confusion_matrix(y_true, y_pred)
        column_width = max([len(str(x)) for x in labels] + [5])  # 5 is value length
        print("Confusion Matrix for " + print_label + ":")
        #report = " " * column_width + " " + "{:_^{}}".format("prediction", column_width * len(labels))+ "\n"
        report = " " * column_width + " ".join(["{:>{}}".format(label, column_width) for label in labels]) + "\n"
        for i, label1 in enumerate(labels):
            report += "{:>{}}".format(label1, column_width) + " ".join(["{:{}d}".format(cm[i, j], column_width) for j in range(len(labels))]) + "\n"
        print(report)

    def __print_cr(self, y_true, y_pred, target_names, print_label):
        print("Classification Report for " + print_label + ":")
        print(classification_report(y_true, y_pred, target_names=target_names))

    def plot_confusion_matrix(self, cm, classes, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def print_roc(self):
        for label, estimator in self.roc_estimators.items():
            estimator.fit(self.data_train, self.target_train)
            proba_for_each_class = estimator.predict_proba(self.data_test)

            fpr, tpr, thresholds = roc_curve(self.target_test, proba_for_each_class[:,1])

            plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

        plt.legend()
        plt.show()

    def classify_nb(self): 
        ''' Classification with Naive Bayes. '''
        cl_label = "Naive Bayes"
        naive_bayes = GaussianNB()
        naive_bayes.fit(self.data_train, self.target_train)
        prediction = naive_bayes.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_nb:
            self.accuracy_nb = acc
        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        #self.plot_confusion_matrix(confusion_matrix(self.target_test, prediction), classes=self.binary_labels)
        self.roc_estimators[cl_label] = naive_bayes
        print('-' * 52)

    def classify_knn(self, n=3, display_matrix=False):
        ''' Classification with K_Nearest_Neighbor. '''
        cl_label = "k-Nearest Neighbors"
        knn_estimator = KNeighborsClassifier(n)
        knn_estimator.fit(self.data_train, self.target_train)
        prediction = knn_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_knn:
            self.accuracy_knn   = acc
            self.accuracy_knn_n = n
        if display_matrix:
            self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
            self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
            print('-' * 52)
        self.roc_estimators[cl_label] = knn_estimator

    def classify_nc(self):
        ''' Classification with Nearest Centroid. '''
        cl_label = "Nearest Centroid"
        nc_estimator = NearestCentroid()
        nc_estimator.fit(self.data_train, self.target_train)
        prediction = nc_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_nc:
            self.accuracy_nc = acc
        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        print('-' * 52)

    def classify_svm(self, display_roc=False): #, C=1.0, gamma ='auto')
        ''' Classification with Support Vector Machine. '''
        cl_label = "Support Vector Machine"
        if not display_roc:
            svm_estimator = SVC()
        else:
            svm_estimator = SVC(probability=True)
            self.roc_estimators[cl_label] = svm_estimator
        svm_estimator.fit(self.data_train, self.target_train)
        prediction = svm_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        if acc > self.accuracy_svm:
            self.accuracy_svm = acc
        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        print('-' * 52)

    def classify_dt(self):
        ''' Clasificiation with Decision Tree'''
        cl_label = "Decision Tree"
        import graphviz
        import subprocess
        import time
        from sklearn.utils.multiclass import unique_labels

        decision_tree = tree.DecisionTreeClassifier(max_depth=5,criterion="entropy",min_samples_split=2)
        decision_tree.fit(self.data_train,self.target_train) 
        prediction = decision_tree.predict(self.data_test)

        dot_data = tree.export_graphviz(decision_tree,
            feature_names=self.data_encoded.columns.values,
            class_names=unique_labels(self.dataset['perceived_quality']),
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None
        )

        # only works on Unix?
        '''
        with open('../data/plots/tree.dot', 'w') as f:
            f.write(dot_data)
        time.sleep(2)
        subprocess.call(['dot -Tpng ' + io.get_universal_path('../data/plots/tree.dot') + ' -o ' + io.get_universal_path('../data/plots/image.png')], shell=True)
        '''
        acc = accuracy_score(self.target_test,prediction) * 100
        if acc > self.accuracy_dt:
            self.accuracy_dt = acc
        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        self.roc_estimators[cl_label] = decision_tree
        print('-' * 52)

    def exclude_amenity_columns(self):
        start = "Amenity_TV"
        end = "Amenity_Paidparkingoffpremises"
        self.__drop(start, end)

    def exclude_transit_tfidf(self):
        # No difference.
        start = "transit_10"
        end = "transit_west"
        self.__drop(start, end)

    def exclude_neighbor_tfidf(self):
        if not self.long_tfidf:
            start = "neighborhood_overview_area"
            end = "neighborhood_overview_walk"
        else:
            start = "neighborhood_overview_10"
            end = "neighborhood_overview_west"
        self.__drop(start, end)

    def exclude_description_tfidf(self):
        if not self.long_tfidf:
            start = "description_access"
            end = "description_walk"
        else:
            start = "description_10"
            end = "description_zone"
        self.__drop(start, end)

    def exclude_houserules_tfidf(self):
        if not self.long_tfidf:
            start = "house_rules_allow"
        else:
            start = "house_rules_10pm"
        end = "house_rules_use"
        self.__drop(start, end)

    # Parameter Tuning SVM
    def para_tuning_SVM(self, loose, fine, use_sample):
        if use_sample:
            print('Test on small Dataset')
            sample = self.data_encoded_with_label.sample(n=1000, random_state=12)
            target_label = sample['perceived_quality']
            data = sample.drop('perceived_quality', axis=1, inplace=False)
        else:
            print('Test on whole Dataset')
            target_label=self.label
            data = self.data_encoded
        if loose:
            print('Lose Grid Search')
            self.loose_grid_search_SVM(data= data, target=target_label)
            #print('Results on small set:\n')
            #print("NOT needed again, best results are  small Dataset = C=2**9 and gamma is 2**-13") #TODO
            #print('Results on whole set:\n')
        if fine:
            print('Fine Grid Search')
            self.fine_grid_search_SVM(data= data, target=target_label)
            #print('Results on small fixed sample set:\n')
            #print ('best score is 0.682 with params C: 512, gamma: 0.0001220703125, kernel: rbf}')
            #print('Results on small sample set:\n')
            #print ('best score is 0.682 with params C: 768, gamma: 0.0001220703125, kernel: rbf}')
            #print('Results on whole set:\n')

    def loose_grid_search_SVM(self, data, target):
        parameters = {
            'kernel':['rbf'],
            'C': [1, 2**(-5), 2**(-3),2**(-1),2**(1),2**(3),2**(5),2**(7),2**(9),2**(11),2**(13),2**(15), 1, 20],# penalty 
            'gamma': [2**(-9),2**(-7),2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5), 'auto']#Kernel coefficient for ‘rbf’=> if ‘auto’ then 1/n_features used
        }

        self.start_gridsearch_svm(parameters, data, target)

    def fine_grid_search_SVM(self, data, target):
        #TODO again after loose Grid Search
        parameters = {
            'kernel': ['rbf'],
            'C': [ 19, 19.5, 20, 20.5, 21],# penalty
            'gamma': [0.0077, 0.00775, 0.0078, 2**(-7),0.00785, 0.0079] # quarter i*1/4 away from middle 2^-13
        }
        self.start_gridsearch_svm(parameters, data, target)

    def start_gridsearch_svm(self, parameters, data, target):
        clf = SVC()
        print('using 3 Fold Cross-Validation')
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy', cv=cv, verbose=100, n_jobs=-1)
        grid_search_estimator.fit(data, target)

        print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))

    def para_tuning_dt(self):
        '''Testing best values'''
        target_label = self.label
        decision_tree =tree.DecisionTreeClassifier()
        parameters = {
            'criterion':['gini', 'entropy'], 
            'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, None],
            'min_samples_split' :[2,3,4,5,6,7,8,9]
        }
        stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search_estimator = GridSearchCV(decision_tree,parameters,scoring = 'accuracy', cv=stratified)
        grid_search_estimator.fit(self.data_encoded,target_label)
        print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))
        
    
    # Parameter Tuning k-NN
    def para_tuning_knn(self):
        clf = KNeighborsClassifier()
        #parameter = clf.get_params()
        #print(parameter)
        target = self.label
        data = self.data_encoded
        self.grid_search_knn(data = data, target=target)

    def grid_search_knn(self, data, target):
        clf = KNeighborsClassifier()
        print('using 10 Fold Cross-Validation for Classifier k-NN')
        parameters = {
            'n_neighbors':[3,4,5,6], 
            'algorithm':['ball_tree', 'kd_tree', 'brute'],
            'p': [1,2],
            'weights': ['uniform', 'distance']
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy', cv=cv, verbose=100, n_jobs=-1)
        grid_search_estimator.fit(data,target)

        print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))


    # Parameter Tuning NC
    def para_tuning_nc(self):
        clf = NearestCentroid()
        #parameter = clf.get_params()
        #print(parameter)
        target = self.label
        data = self.data_encoded
        self.grid_search_nc(data = data, target=target)

    def grid_search_nc(self, data, target):
        clf = NearestCentroid()
        print('using 10 Fold Cross-Validation for Classifier NC')
        parameters = {
            'metric':['euclidean','l2', 'l1', 'manhattan'], 
            'shrink_threshold':[None, 0.1,0.2,0.3,0.4,0.5],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy', cv=cv, verbose=100, n_jobs=-1)
        grid_search_estimator.fit(data,target)

        print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))