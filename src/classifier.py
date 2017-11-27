''' Module containing methods for classification and evaluation of classifiers. '''
import os
#import matplotlib.pyplot as plt

from category_encoders.ordinal import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

#import numpy as np
#from sklearn.metrics import roc_curve
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.model_selection import cross_val_predict
#from scipy import interp
#from sklearn.metrics import roc_curve, auc

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

        #self.clas_rep_nb = "?"
        #self.confusion_ma_nb = "?"
        #self.clas_rep_knn = "?"
        #self.confusion_ma_knn = "?"
        #self.clas_rep_dt = "?"
        #self.confusion_ma_dt = "?"
        #self.clas_rep_svm = "?"
        #self.confusion_ma_svm = "?"
        #self.clas_rep_nc = "?"
        #self.confusion_ma_nc = "?"

        self.long_tfidf = long_tfidf
        self.display_columns = display_columns
        
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
               #"\nClassification Report NB"+ '{0:.2f}%'.format(self.confusion_ma_nb) +
               #"\n\nConfusion Matrix NB"+ '{0:.2f}%'.format(self.clas_rep_nb)

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

    def classify_nb(self): 
        ''' Classification with Naive Bayes. '''
        naive_bayes = GaussianNB()
        naive_bayes.fit(self.data_train, self.target_train)
        prediction = naive_bayes.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        #clas_rep_nb = classification_report(self.target_test, prediction)
        #confusion_ma_nb = confusion_matrix(self.target_test, prediction)
        if acc > self.accuracy_nb:
            self.accuracy_nb = acc

    def classify_knn(self, n=3):
        ''' Classification with K_Nearest_Neighbor. '''
        knn_estimator = KNeighborsClassifier(n)
        knn_estimator.fit(self.data_train, self.target_train)
        prediction = knn_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        #clas_rep_knn = classification_report(self.target_test, prediction)
        #confusion_ma_knn = confusion_matrix(self.target_test, prediction)
        if acc > self.accuracy_knn:
            self.accuracy_knn   = acc
            self.accuracy_knn_n = n

    def classify_nc(self):
        ''' Classification with Nearest Centroid. '''
        nc_estimator = NearestCentroid()
        nc_estimator.fit(self.data_train, self.target_train)
        prediction = nc_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        #clas_rep_nc = classification_report(self.target_test, prediction)
        #confusion_ma_nc = confusion_matrix(self.target_test, prediction)
        if acc > self.accuracy_nc:
            self.accuracy_nc = acc

    def classify_svm(self): #, C=1.0, gamma ='auto')
        ''' Classification with Support Vector Machine. '''
        svm_estimator = SVC()
        svm_estimator.fit(self.data_train, self.target_train)
        prediction = svm_estimator.predict(self.data_test)
        acc = accuracy_score(self.target_test, prediction) * 100
        #clas_rep_svm = classification_report(self.target_test, prediction)
        #confusion_ma_svm = confusion_matrix(self.target_test, prediction)
        if acc > self.accuracy_svm:
            self.accuracy_svm = acc

    def classify_dt(self):
        ''' Clasificiation with Decision Tree'''
        #import graphviz
        import subprocess
        import time
        from sklearn.utils.multiclass import unique_labels

        decision_tree = tree.DecisionTreeClassifier(max_depth=3,criterion="gini")
        decision_tree.fit(self.data_train,self.target_train) 
        prediction = decision_tree.predict(self.data_test)
        '''
        dot_data = tree.export_graphviz(decision_tree,
            feature_names=self.data_encoded.columns.values,
            class_names=unique_labels(self.dataset['perceived_quality']),
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None
        )
        '''
        #with open('../data/plots/tree.dot', 'w') as f:
        #    f.write(dot_data)
        #time.sleep(2)

        #subprocess.call(['dot -Tpng ../data/plots/tree.dot -o ../data/plots/image.png'], shell=True)

        acc = accuracy_score(self.target_test,prediction) * 100
        #clas_rep_dt = classification_report(self.target_test, prediction)
        #confusion_ma_dt = confusion_matrix(self.target_test, prediction)
        if acc > self.accuracy_dt:
            self.accuracy_dt = acc

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

    def __drop(self, start, end):
        self.data_encoded.drop(self.data_encoded.columns[self.data_encoded.columns.get_loc(start): self.data_encoded.columns.get_loc(end) + 1], axis=1, inplace=True)

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

        if (loose):
            print('Lose Grid Search')
            self.loose_grid_search_SVM(data= self.data_encoded, target=target_label)
        if (fine):
            print('Fine Grid Search')
            self.fine_grid_search_SVM(data, target=target_label)

    def loose_grid_search_SVM(self, data, target):
        parameters = {
            'kernel':[ 'rbf'],
            'C': [2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5),2**(7),2**(9),2**(11),2**(13),2**(15), 1, 20],# penalty 
            'gamma': [2**(-15),2**(-13),2**(-11),2**(-9),2**(-7),2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5), 'auto']#Kernel coefficient for ‘rbf’=> if ‘auto’ then 1/n_features used
        }

        self.start_GridSearch_SVM(parameters, data, target)

    def fine_grid_search_SVM(self, data, target):
        #TODO again after loose Grid Search
        parameters = {
            'kernel': ['rbf'],
            'C': [0.1, 0.2, 0.3, 2**(-5), 0.4, 0.5, 0.6, 0.7],# penalty 
            'gamma': [2**(-17),2**(-16),2**(-15),2**(-14),2**(-13),2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7)]
        }
        self.start_GridSearch_SVM(parameters, data, target)

    def start_GridSearch_SVM(self,parameters, data, target):
        clf = SVC()
        print('using 10 Fold Cross-Validation')
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy', cv=cv)
        grid_search_estimator.fit(data, target)

        print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))

        #def roc_curve(self, n=5):
    #    knn_estimator = KNeighborsClassifier(n)
    #    knn_estimator.fit(self.data_train, self.target_train)
    #    proba_for_each_class = knn_estimator.predict_proba(self.data_test)
    #    fpr, tpr, thresholds = roc_curve(self.target_test, proba_for_each_class[:,1], pos_label='good')

    #    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    #    plt.plot(fpr,tpr,label='K-NN')

    #    plt.legend()
    #    plt.show() 

    #def prediction(self):
    #    methods = [[decision_tree, svm_estimator, nc_estimator, naive_bayes, knn_estimator]]
    #    for method in methods:
    #        predicted = cross_val_predict(method, data_encoded, target, cv = cv)
    #        print("predicted value {} with method {}".format(predicted, method))

    #def postprocess(self):
        #data_target = np.array([x.decode('ascii') for x in self.data_encoded['perceived_quality'].values])
        #data_data = pd.get_dummies(self.data_encoded.drop('perceived_quality', axis=1))
        #print(data_data.head())
        #print(data_target.head())

    #def avg_roc(cv, estimator, data, target, pos_label):
    #    mean_fpr = np.linspace(0, 1, 100)
    #    tprs = []
    #    aucs = []    
    #    for train_indices, test_indices in cv.split(data, target):
    #        train_data, train_target = data[train_indices], target[train_indices]
    #        estimator.fit(train_data, train_target)
        
    #        test_data, test_target = data[test_indices], target[test_indices]
    #        decision_for_each_class = estimator.predict_proba(test_data) 
    
    #        fpr, tpr, thresholds = roc_curve(test_target, decision_for_each_class[:,1], pos_label=pos_label)
    #        tprs.append(interp(mean_fpr, fpr, tpr))
    #        tprs[-1][0] = 0.0 
    #        aucs.append(auc(fpr, tpr))        
        
    #    mean_tpr = np.mean(tprs, axis=0)
    #    mean_tpr[-1] = 1.0
    #    mean_auc = auc(mean_fpr, mean_tpr)
    #    std_auc = np.std(aucs)    
    #    return mean_fpr, mean_tpr, mean_auc, std_auc

    #def plot_roc_curve(self):
    #    mean_fpr, mean_tpr, mean_auc, std_auc = avg_roc(cv, knn_estimator, data_data.values, data_target, 'Good')
    #    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    #    plt.plot(mean_fpr,mean_tpr,label='K-NN')

    #    plt.legend()
    #    plt.show()


