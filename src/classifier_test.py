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
def decision():
    df =io.read_csv('../data/playground/encoded.csv')
    score_target_binned = df['perceived_quality']
    score_data = df.drop(columns=['id','perceived_quality']) 

    #Decision Tree plot
    decision_tree = tree.DecisionTreeClassifier(max_depth=3,criterion="gini")
    decision_tree.fit(score_data,score_target_binned) 
    dot_data = tree.export_graphviz(decision_tree,feature_names=score_data.columns.values,out_file=None,filled=True,rounded=True)   
    graph = pydotplus.graph_from_dot_data(dot_data)
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)
    # write in png file
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
 
    for edge in edges:
        edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
 
    graph.write_png('../data/playground/tree.png')

    # (Nico): für Dennis: Das ist der Code der bei mir damals funktioniert hat um das als graph mit graphviz zu exportieren. falls du es nicht brauchst, einfach löschen.
    '''decision_tree = tree.DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5)
    decision_tree.fit(iris_binned_and_encoded, iris['Name'])

    dot_data = tree.export_graphviz(decision_tree,
        feature_names=iris_binned_and_encoded.columns.values,
        class_names=unique_labels(iris['Name']),
        filled=True,
        rounded=True,
        special_characters=True,
        out_file=None
    )

    with open('my_dot.dot', 'w') as f:
        f.write(dot_data)'''

    #10 Cross-Validation
    data_train, data_test,target_train,target_test =train_test_split(score_data,score_target_binned,test_size=0.3,random_state=42, stratify= score_target_binned)
    print(data_train.head())
    accuracy_rating = cross_val_score(decision_tree,score_data,score_target_binned,cv = 10, scoring ='accuracy')
    print(accuracy_rating.mean())
  
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
nearest_centroid(dataset)


