import pandas as pd
import io_util as io
import numpy as np
import graphviz
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels

def naive_bayes(dataset):
    '''Trains and tests a Naive Bayes Classifier with selected features'''
    ###können wir das nicht für alle methoden verwenden bis zu data_train, data_test,...?
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from category_encoders.ordinal import OrdinalEncoder
    from sklearn.metrics import accuracy_score
    
    encoder = OrdinalEncoder()
    dataset_encoded = encoder.fit_transform(dataset)
    listings_data=dataset_encoded.drop(columns=['id','perceived_quality'])
    listings_target= dataset_encoded['perceived_quality']
    
    data_train, data_test, target_train, target_test = train_test_split(listings_data, listings_target, test_size=0.2, random_state=42, stratify=listings_target)
    
    naive_bayes = GaussianNB()
    naive_bayes.fit(data_train, target_train)
    prediction = naive_bayes.predict(data_test)

    #nbresults=pd.DataFrame(data_test)
    #quality_predicted = nbresults.assign(predicted_quality=prediction)
    #io.write_csv(quality_predicted, '../data/playground/naivebayes.csv')
    
    accuracy=accuracy_score(target_test, prediction)
    print('Accuracy of Naive Bayes Classifier:{}'.format(accuracy))

dataset = io.read_csv('../data/playground/dataset.csv')
naive_bayes(dataset)

# Not yet done, refer to lecture slide "Pictures": Gini possible with continous+categoical values
def decison():
  
  df =io.read_csv('../data/playground/dataset.csv')
  df = df.dropna()
  score_target_binned = df['perceived_quality']
  score_data = df[['bathrooms','bedrooms','beds','host_location']]
 # score_data = df[['experiences_offered','host_name','host_since','host_location','host_response_time','host_response_rate','host_is_superhost','host_verifications','host_has_profile_pic','host_identity_verified','neighbourhood_cleansed','latitude','longitude','is_location_exact','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','square_feet','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','availability_30','availability_60','availability_90','availability_365','number_of_reviews','first_review','last_review']]
  #score_data = score_data.drops('review_scores_rating',axis =1)
  #bins = [60,75,100] #Example range for review_scores_rating
  #score_target_binned = pd.DataFrame(dict (review_scores_rating = pd.cut(df["review_scores_rating"],bins=3,labels =['low','middle','high'])))
  #Decision Tree plot
  decision_tree = tree.DecisionTreeClassifier(max_depth=3,criterion="gini")
  decision_tree.fit(score_data,score_target_binned) #ggf. paramter durch testdaten tauschen
 # with open('../data/playground/test.txt') as f:
  #    tree.export_graphviz(decision_tree, feature_names = score_data.columns.values, class_names= unique_labels(score_target_binned), filled=True,rounded=True, special_characters=True, out_file = f)


  dot_data = tree.export_graphviz(decision_tree, feature_names = score_data.columns.values, class_names= unique_labels(score_target_binned), filled=True,rounded=True, special_characters=True, out_file = None)
  graphviz.Source(dot_data)
  print(decision_tree.tree_.node_count) 
  #10 Cross-Validation
  data_train, data_test,target_train,target_test =train_test_split(score_data,score_target_binned,test_size=0.3,random_state=42, stratify= score_target_binned)
  print(data_train.head())
  accuracy_rating = cross_val_score(decision_tree,score_data,score_target_binned,cv = 10, scoring ='accuracy')
  print(accuracy_rating.mean())
  
def knn(dataset, knn_estimator, data_train, target_train, data_test, target_test):
    '''KNN'''
    knn_estimator = KNeighborsClassifier(4)
    knn_estimator.fit(data_train, target_train)

    predict = knn_estimator.predict(data_test)
    print('Prediction of KNN Classifier:{}'.format(predict))

    accuracy = knn_estimator.score(target_test, predict)
    print('Accuracy of KNN Classifier:{}'.format(accuracy))
