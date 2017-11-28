''' Module containing methods for classification and evaluation of classifiers. '''
import os
import matplotlib.pyplot as plt

import numpy as np
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
        
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None
        
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
        self.data_encoded.drop('perceived_quality', axis=1, inplace=True)
        self.data_encoded.drop('id', axis=1, inplace=True)

        # Insert drop of single columns here
        #self.data_encoded.drop(['instant_bookable', 'require_guest_profile_picture', 'first_review', 'last_review'], axis=1, inplace=True)

        # Insert drop of tfidf/amenity columns here
        self.exclude_amenity_columns()
        #self.exclude_transit_tfidf()
        #self.exclude_description_tfidf()
        #self.exclude_houserules_tfidf()
        self.exclude_neighbor_tfidf()

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

        decision_tree = tree.DecisionTreeClassifier(max_depth=3,criterion="gini")
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
        with open('../data/plots/tree.dot', 'w') as f:
            f.write(dot_data)
        time.sleep(2)

        subprocess.call(['dot -Tpng ' + io.get_universal_path('../data/plots/tree.dot') + ' -o ' + io.get_universal_path('../data/plots/image.png')], shell=True)

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
    def para_tuning_SVM(self, loose, fine):
        # TODO
        # on whole data set or only on training? 
        # - save path of final file
        self.data_encoded = io.read_csv('../data/final/dataset_2_encoded.csv')
        # we need the whole dataset or not?
        target_label = self.data_encoded['perceived_quality'] 
        self.data_encoded.drop('perceived_quality', axis=1, inplace=True)
        self.data_encoded.drop('id', axis=1, inplace=True)

        print('Test on small Dataset')
        target_label = target_label[:1000]
        data = self.data_encoded[:1000]

        if loose:
            print('Lose Grid Search')
            self.loose_grid_search_SVM(data= self.data_encoded, target=target_label)
        if fine:
            print('Fine Grid Search')
            self.fine_grid_search_SVM(data, target=target_label)

    def loose_grid_search_SVM(self, data, target):
        parameters = {
            'kernel':[ 'rbf'],
            'C': [2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5),2**(7),2**(9),2**(11),2**(13),2**(15), 1, 20],# penalty 
            'gamma': [2**(-15),2**(-13),2**(-11),2**(-9),2**(-7),2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5), 'auto']#Kernel coefficient for ‘rbf’=> if ‘auto’ then 1/n_features used
        }

        self.start_gridsearch_svm(parameters, data, target)

    def fine_grid_search_SVM(self, data, target):
        #TODO again after loose Grid Search
        parameters = {
            'kernel': ['rbf'],
            'C': [0.1, 0.2, 0.3, 2**(-5), 0.4, 0.5, 0.6, 0.7],# penalty 
            'gamma': [2**(-17),2**(-16),2**(-15),2**(-14),2**(-13),2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7)]
        }
        self.start_gridsearch_svm(parameters, data, target)

    def start_gridsearch_svm(self, parameters, data, target):
        clf = SVC()
        print('using 10 Fold Cross-Validation')
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy', cv=cv)
        grid_search_estimator.fit(data, target)

        print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))