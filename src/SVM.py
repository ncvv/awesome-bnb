import pandas as pd
import io_util as io
import warnings
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from category_encoders.ordinal import OrdinalEncoder
import category_encoders as ce
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

def svm(dataset):  
    cols = dataset.columns   
    #print ((cols))

    target_label = dataset['perceived_quality']
    #TODO split DataFrame
    idx_amenity = dataset.columns.get_loc("Amenity_TV")
    idx_amenity_end = dataset.columns.get_loc ("Amenity_Paidparkingoffpremises")
   
    data = dataset.drop(['id', 'perceived_quality'],axis=1)
    #print(data.head())


    #drop tables ACC using one way
    # before Accuracy SVM:0.5301204819277109
    # now SVM:0.5308734939759037
    data = data.drop([ 'availability_60', 'availability_90', 'availability_365', 'availability_30', 'Amenity_Pool', 'Amenity_Freeparkingonstreet',  'Amenity_Paidparkingoffpremises', 'calculated_host_listings_count', 'Amenity_Dog(s)',  'Amenity_Cat(s)' , 'Amenity_Freeparkingonpremises', 'Amenity_Elevatorinbuilding',  'Amenity_Petsallowed', 'Amenity_Internet','minimum_nights' ,'Amenity_Shampoo', 'Amenity_24-hourcheck-in',  ],axis=1) 

    #one_way(data, target_label)
    print('Parameter tuning')
   # using_GridSearch(data, target_label, use_small_dataset=True, use_cv=True, loose=True,  fine =False)
    #using_GridSearch(data, target_label, use_small_dataset=True, use_cv=True, loose=False,  fine =True)
    #using_GridSearch(data, target_label, use_small_dataset=True, use_cv=True, loose=True,  fine =False)
    #using_GridSearch(data, target_label, use_small_dataset=True, use_cv=True, loose=False,  fine =True)
    
    
    #print('Test derived Parameters')
    #using_GridSearch(data, target_label, use_small_dataset=False, use_cv=True, loose=False,  fine =False)
    
def one_way(data, target_label):
    clf = SVC(kernel='rbf')#, decision_function_shape='ovr')
    #data = data.values.reshape(1,-1)
    print(data.shape)
    #print(len(target_label.values))
    
    data_train, data_test, target_train, target_test =  split_dataset_regular(data, target_label) #, test_size=0.2, random_state=42, stratify=target_label)

    clf.fit(data_train, target_train)
    pred = clf.predict(data_test)
    acc=accuracy_score(target_test, pred)
    #clf.score(target_test, pred)
    print('Accuracy SVM:{}'.format(acc))
    #print([pred])
    #print(pd.DataFrame(pred).shape)
    #TODO hier wird ein FEHLER geworfen muss ein [[]] sein statt []
    #print(clf.score(target_test, pred.tolist))
    #print('SVM score:{}'.format(clf.score))

def using_GridSearch(data, target_label, use_small_dataset,  use_cv , loose, fine):
    #make it smaller for testing
    if(use_small_dataset):
        print('Test on small Dataset')
        target_label = target_label[:1000]
        data = data[:1000]
        if (loose):
            print('Lose Grid Search')
            loose_grid_search(data, target_label, use_cv)
        if (fine):
            print('Fine Grid Search')
            fine_grid_search(data, target_label, use_cv)

    else:
        print('Test on whole Dataset')
        C=0.6
        gamma = 2**(-16)
        clf = SVC(kernel='rbf', C=C, gamma=gamma)
        data_train, data_test, target_train, target_test =  split_dataset_regular(data, target_label) 
        clf.fit(data_train, target_train)
        pred = clf.predict(data_test)
        acc=accuracy_score(target_test, pred)
        #clf.score(target_test, pred)
        print('Accuracy SVM:{}'.format(acc))

    
    

def loose_grid_search(data, target, use_cv):
    parameters = {
        'kernel':[ 'rbf'],
        'C': [2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5),2**(7),2**(9),2**(11),2**(13),2**(15), 1, 20],# penalty 
        'gamma': [2**(-15),2**(-13),2**(-11),2**(-9),2**(-7),2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5), 'auto']#Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
        #'decision_function_shape': ['ovo', 'ovr'] # ovr recommended
    }

    start_GridSearch(parameters, data, target, use_cv)
    


def start_GridSearch(parameters, data, target, use_cv):
    clf = SVC()
    if (use_cv):
        print('using Cross-Validation')
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10 is a good number empirical studies
        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy', cv=cv)
    else:
        print('NO Cross-Validation')
        grid_search_estimator = GridSearchCV(clf, parameters, scoring='accuracy')

    grid_search_estimator.fit(data, target)

    print("best score is {} with params {}".format(grid_search_estimator.best_score_, grid_search_estimator.best_params_ ))
    '''
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
       print(results['mean_test_score'][i])
       # print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))
    '''
  

def fine_grid_search(data, target, use_cv):
    parameters = {
        'kernel': ['rbf'],
        'C': [0.1, 0.2, 0.3, 2**(-5), 0.4, 0.5, 0.6, 0.7],# penalty 
        'gamma': [2**(-17),2**(-16),2**(-15),2**(-14),2**(-13),2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7)]
    }

    start_GridSearch(parameters, data, target, use_cv)


def encode_whole_dataset(dataset, save, filename):
    from category_encoders.ordinal import OrdinalEncoder
    import pandas as pd
    import io_util as io

    encoder = OrdinalEncoder()
    data_encoded = encoder.fit_transform(dataset)
    print(data_encoded.head())
    if (save):
        path = '../data/playground/' + filename + '.csv'
        print (path)
        io.write_csv(data_encoded, path )
    


def init ():
    listings = io.read_csv('../data/playground/dataset.csv')
    encode_whole_dataset(listings, True, 'encoded_listings')

def init_onehot ():
    from category_encoders.ordinal import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import io_util as io    
    listings = io.read_csv('../data/playground/dataset.csv')
    encoder = OneHotEncoder()
    data_encoded = encoder.fit_transform(listings)
    print(data_encoded.head())
    
    path = '../data/playground/' + filename + '_onehot.csv'
    print (path)
    io.write_csv(data_encoded, path )

def split_dataset_specific(data, target, test_size, stratify_target):
    if (stratify_target):
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_size, random_state=42, stratify=target)
    else:
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_size, random_state=42)
    return data_train, data_test, target_train, target_test

def split_dataset_regular(data, target):
    return split_dataset_specific(data, target, test_size=0.2, stratify_target=True)

def accuracy_per_column(cols, dataset):
    for column in cols:
        print(column)
        attr = dataset[str(column)]
        attr = attr.values.reshape(-1,1) # only needed if there is  one attribute
        target_label = dataset['perceived_quality']
        one_way(attr, target_label)

def test_C_gamma(data, target, C, gamma):
    print('Test on whole Dataset')
    #clf_all= SVC(kernel='rbf', C=2**(-5), gamma='auto')    


#init()
#init_onehot()

encoded_listings = listings = io.read_csv('../data/playground/encoded_listings.csv')
svm (encoded_listings)
#print('Done')
#svm(listings)


'''
TODO #
# sklearn. Dummy Classifier als BASELINE 
# balance, wenn es was bringt - sieht man an der Confusion Matrix 
# TFID - Prune [NICO]
# REGEX  - um text features aus dem Text zu ziehen
>> Wichtig: Sachen begründen, zeigen, dass man es verstanden hat
plots zeigen Vorteil Python
PAPER SVM : http://www.datascienceassn.org/sites/default/files/Practical%20Guide%20to%20Support%20Vector%20Classification.pdf

# '''
