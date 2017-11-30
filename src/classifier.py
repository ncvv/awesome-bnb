''' Module containing methods for classification and evaluation of classifiers. '''
import os
import itertools
import time
import subprocess

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.utils.multiclass import unique_labels
from category_encoders.ordinal import OrdinalEncoder

import io_util as io

class Classifier(object):
    ''' Class for classification. '''

    def __init__(self, dataset, path, long_tfidf, display_columns, scoring):
        self.dataset = dataset
        self.label = None
        self.data_encoded = None
        self.data_encoded_with_label = None

        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None

        self.accuracy_knn = 0
        self.accuracy_knn_n = "?"
        self.accuracy_nb = 0
        self.accuracy_mnb=0
        self.accuracy_dt = 0
        self.accuracy_svm = 0
        self.accuracy_nc = 0

        self.scoring = scoring

        self.binary_labels = ["Bad", "Good"]
        self.long_tfidf = long_tfidf
        self.display_columns = display_columns
        self.roc_estimators = {}

        self.encode_and_split(path)

    #toString() equivalent to Java.
    def __str__(self):
        return ("Classification Results:"
               "\nAccuracy NB:  " + '{0:.2f}%'.format(self.accuracy_nb) +
               "\nAccuracy MNB: " + '{0:.2f}%'.format(self.accuracy_mnb) +
               "\nAccuracy kNN: " + '{0:.2f}%'.format(self.accuracy_knn) + ', n=' + str(self.accuracy_knn_n) +
               "\nAccuracy DT:  " + '{0:.2f}%'.format(self.accuracy_dt) +
               "\nAccuracy SVM: " + '{0:.2f}%'.format(self.accuracy_svm) +
               "\nAccuracy NC:  " + '{0:.2f}%'.format(self.accuracy_nc) +
               "\n\nMax.: " + '{0:.2f}%'.format(max(self.accuracy_nb, self.accuracy_mnb, self.accuracy_knn, self.accuracy_dt, self.accuracy_svm, self.accuracy_nc)))

    def encode_and_split(self, path):
        ''' Encode datatset and split it into training and test data. '''
        if os.path.exists(path):
            self.data_encoded = io.read_csv(path)
        else:
            encoder = OrdinalEncoder()
            self.data_encoded = encoder.fit_transform(self.dataset)
            print('Encoding done for file: ' + str(path))
            io.write_csv(self.data_encoded, path)

        self.label = self.data_encoded['perceived_quality']

        # Insert drop of single columns here
        drop_list = ['id', 'perceived_quality', 'instant_bookable', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'host_location', 'host_verification_binned', 'host_response_rate_binned', 'require_guest_phone_verification', 'host_response_time', 'zipcode', 'require_guest_profile_picture', 'calculated_host_listings_count', 'first_review', 'last_review']
        self.data_encoded.drop(drop_list, axis=1, inplace=True)
        self.data_encoded_with_label = pd.concat([self.data_encoded, self.label], axis=1)

        # Insert drop of tfidf/amenity columns here
        self.exclude_amenity_columns()
        #self.exclude_transit_tfidf()
        #self.exclude_description_tfidf()
        #self.exclude_houserules_tfidf()
        self.exclude_neighbor_tfidf()

        if self.display_columns:
            print('Columns:\n' + '\n'.join(list(self.data_encoded)) + '\n')

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(self.data_encoded, self.label, test_size=0.2, random_state=42, stratify=self.label)

    def __drop(self, start, end):
        ''' Drop columns from data_encoded at given indices. '''
        self.data_encoded.drop(self.data_encoded.columns[self.data_encoded.columns.get_loc(start): self.data_encoded.columns.get_loc(end) + 1], axis=1, inplace=True)

    def __print_cm(self, y_true, y_pred, labels, print_label):
        ''' Print confusion matrix. '''
        cm = confusion_matrix(y_true, y_pred)
        column_width = max([len(str(x)) for x in labels] + [5])  # 5 is value length
        print("Confusion Matrix for " + print_label + ":")
        report = " " * column_width + " ".join(["{:>{}}".format(label, column_width) for label in labels]) + "\n"
        for i, label1 in enumerate(labels):
            report += "{:>{}}".format(label1, column_width) + " ".join(["{:{}d}".format(cm[i, j], column_width) for j in range(len(labels))]) + "\n"
        print(report)

    def __print_cr(self, y_true, y_pred, target_names, print_label):
        ''' Print classification report. '''
        print("Classification Report for " + print_label + ":")
        print(classification_report(y_true, y_pred, target_names=target_names))

    def __print_bs(self, gse):
        ''' Print best accuracy score of cross validation with gridsearch. '''
        print("Best Score is {} with Parameters {}".format(gse.best_score_, gse.best_params_))

    def __plot_cm(self, cm, classes, print_label, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues, show=True):
        ''' Plot confusion matrix and save it as .pdf '''
        plt.clf()
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
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig('../data/plots/' + print_label + '_CM.pdf', bbox_inches='tight')
        if show:
            plt.show()

    def plot_roc(self, show=True):
        ''' Plot ROC curve of classification results and save it as .pdf '''
        plt.clf()
        
        for label, estimator in self.roc_estimators.items():
            estimator.fit(self.data_train, self.target_train)
            proba_for_each_class = estimator.predict_proba(self.data_test)

            fpr, tpr, thresholds = roc_curve(self.target_test, proba_for_each_class[:, 1])

            plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.savefig('../data/plots/ROC.pdf', bbox_inches='tight')
        if show:
            plt.show()

    def classify_nb(self, cross_validate=False): 
        ''' Classification with Gaussian Naive Bayes. '''
        cl_label = "Gaussian Naive Bayes"
        naive_bayes = GaussianNB()
        naive_bayes.fit(self.data_train, self.target_train)
        prediction = naive_bayes.predict(self.data_test)

        if cross_validate:
            self.accuracy_nb = float("{:.4f}".format(cross_val_score(naive_bayes, self.data_encoded, self.label, cv=10, scoring=self.scoring).mean())) * 100
        else:
            acc = accuracy_score(self.target_test, prediction) * 100
            if acc > self.accuracy_nb:
                self.accuracy_nb = acc

        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        self.__plot_cm(confusion_matrix(self.target_test, prediction), classes=self.binary_labels, print_label=cl_label, show=False)
        self.roc_estimators[cl_label] = naive_bayes

    def classify_mnb(self, cross_validate=False): 
        ''' Classification with Multinomial Naive Bayes. '''
        cl_label = "Multinomial Naive Bayes"
        multi_naive_bayes = MultinomialNB(alpha=1.0e-10)
        multi_naive_bayes.fit(self.data_train, self.target_train)
        prediction = multi_naive_bayes.predict(self.data_test)

        if cross_validate:
            self.accuracy_mnb = float("{:.4f}".format(cross_val_score(multi_naive_bayes, self.data_encoded, self.label, cv=10, scoring=self.scoring).mean())) * 100
        else:
            acc = accuracy_score(self.target_test, prediction) * 100
            if acc > self.accuracy_mnb:
                self.accuracy_mnb = acc

        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        self.__plot_cm(confusion_matrix(self.target_test, prediction), classes=self.binary_labels, print_label=cl_label, show=False)
        self.roc_estimators[cl_label] = multi_naive_bayes

    def classify_knn(self, n, cross_validate=False, display_matrix=False):
        ''' Classification with K_Nearest_Neighbor. '''
        cl_label = "k-Nearest Neighbors"
        knn_estimator = KNeighborsClassifier(algorithm='ball_tree', p=2, n_neighbors=n, weights='distance')
        knn_estimator.fit(self.data_train, self.target_train)
        prediction = knn_estimator.predict(self.data_test)

        if cross_validate:
            self.accuracy_knn = float("{:.4f}".format(cross_val_score(knn_estimator, self.data_encoded, self.label, cv=10, scoring=self.scoring).mean())) * 100
        else:
            acc = accuracy_score(self.target_test, prediction) * 100
            if acc > self.accuracy_knn:
                self.accuracy_knn = acc
                self.accuracy_knn_n = n

        if display_matrix:
            self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
            self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
            self.__plot_cm(confusion_matrix(self.target_test, prediction), classes=self.binary_labels, print_label=cl_label, show=False)
        self.roc_estimators[cl_label] = knn_estimator

    def classify_nc(self, cross_validate=False):
        ''' Classification with Nearest Centroid. '''
        cl_label = "Nearest Centroid"
        nc_estimator = NearestCentroid(metric='euclidean', shrink_threshold=6)
        nc_estimator.fit(self.data_train, self.target_train)
        prediction = nc_estimator.predict(self.data_test)
        
        if cross_validate:
            self.accuracy_nc = float("{:.4f}".format(cross_val_score(nc_estimator, self.data_encoded, self.label, cv=10, scoring=self.scoring).mean())) * 100
        else:
            acc = accuracy_score(self.target_test, prediction) * 100
            if acc > self.accuracy_nc:
                self.accuracy_nc = acc

        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        self.__plot_cm(confusion_matrix(self.target_test, prediction), classes=self.binary_labels, print_label=cl_label, show=False)
        #self.roc_estimators[cl_label] = nc_estimator

    def classify_svm(self, display_roc=False, cross_validate=False): #, C=1.0, gamma ='auto')
        ''' Classification with Support Vector Machine. '''
        cl_label = "Support Vector Machine"
        if not display_roc:
            svm_estimator = SVC(gamma=0.0078125, kernel='rbf', C=19.5)
        else:
            svm_estimator = SVC(gamma=0.0078125, kernel='rbf', C=19.5, probability=True)
            self.roc_estimators[cl_label] = svm_estimator
        svm_estimator.fit(self.data_train, self.target_train)
        prediction = svm_estimator.predict(self.data_test)

        if cross_validate:
            self.accuracy_svm = float("{:.4f}".format(cross_val_score(svm_estimator, self.data_encoded, self.label, cv=10, scoring=self.scoring).mean())) * 100
        else:
            acc = accuracy_score(self.target_test, prediction) * 100
            if acc > self.accuracy_svm:
                self.accuracy_svm = acc

        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        self.__plot_cm(confusion_matrix(self.target_test, prediction), classes=self.binary_labels, print_label=cl_label, show=False)

    def classify_dt(self, cross_validate=False):
        ''' Clasificiation with Decision Tree. '''
        cl_label = "Decision Tree"
        decision_tree = tree.DecisionTreeClassifier(max_depth=5, criterion="entropy", min_samples_split=6)
        decision_tree.fit(self.data_train, self.target_train) 
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
        if os.name == 'posix':
            with open('../data/plots/tree.dot', 'w') as f:
                f.write(dot_data)
            time.sleep(2)
            subprocess.call(['dot -Tpng ' + io.get_universal_path('../data/plots/tree.dot') + ' -o ' + io.get_universal_path('../data/plots/Decision Tree.png')], shell=True)
            subprocess.call(['rm -f ' + io.get_universal_path('../data/plots/tree.dot')], shell=True)

        if cross_validate:
            self.accuracy_dt = float("{:.4f}".format(cross_val_score(decision_tree, self.data_encoded, self.label, cv=10, scoring=self.scoring).mean())) * 100
        else:
            acc = accuracy_score(self.target_test, prediction) * 100
            if acc > self.accuracy_dt:
                self.accuracy_dt = acc

        self.__print_cm(self.target_test, prediction, labels=self.binary_labels, print_label=cl_label)
        self.__print_cr(self.target_test, prediction, target_names=self.binary_labels, print_label=cl_label)
        self.__plot_cm(confusion_matrix(self.target_test, prediction), classes=self.binary_labels, print_label=cl_label, show=False)
        self.roc_estimators[cl_label] = decision_tree

    def exclude_amenity_columns(self):
        ''' Drop amenity columns from dataset. '''
        start = "Amenity_TV"
        end = "Amenity_Paidparkingoffpremises"
        self.__drop(start, end)

    def exclude_transit_tfidf(self):
        ''' Drop columns for transit TFIDF vector, depending on long_tfidf dataset or normal dataset. '''
        # No difference.
        start = "transit_10"
        end = "transit_west"
        self.__drop(start, end)

    def exclude_neighbor_tfidf(self):
        ''' Drop columns for neighborhood_overview TFIDF vector, depending on long_tfidf dataset or normal dataset. '''
        if not self.long_tfidf:
            start = "neighborhood_overview_area"
            end = "neighborhood_overview_walk"
        else:
            start = "neighborhood_overview_10"
            end = "neighborhood_overview_west"
        self.__drop(start, end)

    def exclude_description_tfidf(self):
        ''' Drop columns for description TFIDF vector, depending on long_tfidf dataset or normal dataset. '''
        if not self.long_tfidf:
            start = "description_access"
            end = "description_walk"
        else:
            start = "description_10"
            end = "description_zone"
        self.__drop(start, end)

    def exclude_houserules_tfidf(self):
        ''' Drop columns for house_rules TFIDF vector, depending on long_tfidf dataset or normal dataset. '''
        if not self.long_tfidf:
            start = "house_rules_allow"
        else:
            start = "house_rules_10pm"
        end = "house_rules_use"
        self.__drop(start, end)

    def tune(self, classifier, parameters, n_splits, data, target):
        ''' Tune parameter values for classifier with cross validation. '''
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid_search_estimator = GridSearchCV(classifier, parameters, scoring='accuracy', cv=cv, verbose=100, n_jobs=-1)
        grid_search_estimator.fit(data, target)
        self.__print_bs(grid_search_estimator)

    def para_tuning_svm(self, loose, fine, use_sample):
        ''' Parameter Tuning SVM. '''
        if loose and fine or not loose and not fine:
            raise ValueError('Exactly one value of loose and fine must be true.')
        
        if use_sample:
            sample = self.data_encoded_with_label.sample(n=1000, random_state=12)
            target = sample['perceived_quality']
            data = sample.drop('perceived_quality', axis=1, inplace=False)
        else:
            target = self.label
            data = self.data_encoded
        # Loose Grid Search
        if loose:
            parameters = {
                'kernel': ['rbf'],
                'C': [1, 2**(-5), 2**(-3), 2**(-1), 2**(1), 2**(3), 2**(5), 2**(7), 2**(9), 2**(11), 2**(13), 2**(15), 1, 20], # penalty 
                'gamma': [2**(-9), 2**(-7), 2**(-5), 2**(-3), 2**(-1), 2**(1), 2**(3), 2**(5), 'auto'] # Kernel coefficient for ‘rbf’ => if ‘auto’ then 1/n_features used
            }
        # Fine Grid Search
        elif fine:
            parameters = {
                'kernel': ['rbf'],
                'C': [19, 19.5, 20, 20.5, 21], # penalty
                'gamma': [0.0077, 0.00775, 0.0078, 2**(-7), 0.00785, 0.0079] # quarter i*1/4 away from middle 2^-13
            }
        self.tune(SVC(), parameters, 3, data=data, target=target)

    def para_tuning_dt(self):
        ''' Parameter Tuning DT. '''
        parameters = {
            'criterion':['gini', 'entropy'], 
            'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, None],
            'min_samples_split' :[2, 3, 4, 5, 6, 7, 8, 9]
        }
        self.tune(tree.DecisionTreeClassifier(), parameters, 3, self.data_encoded, self.label)

    def para_tuning_knn(self):
        ''' Parameter Tuning kNN. '''
        parameters = {
            'n_neighbors':[3,4,5,6], 
            'algorithm':['ball_tree', 'kd_tree', 'brute'],
            'p': [1,2],
            'weights': ['uniform', 'distance']
        }
        self.tune(KNeighborsClassifier(), parameters, 3, self.data_encoded, self.label)

    def para_tuning_nc(self):
        ''' Parameter Tuning NC. '''
        parameters = {
            'metric':['euclidean','manhattan'], 
            'shrink_threshold':[None, 1, 1.1, 6, 7, 8, 9, 10, 11, 12, 13],
        }
        self.tune(NearestCentroid(), parameters, 3, self.data_encoded, self.label)
