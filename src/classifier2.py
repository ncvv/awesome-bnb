import pandas as pd
import io_util as io

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from category_encoders.ordinal import OrdinalEncoder
from sklearn.metrics import accuracy_score

dataset = io.read_csv('../data/playground/dataset.csv')

class Classifier(object):

    def __init__(self, dataset, dataset_encoded, data_train, data_test, target_train, target_test, knn_estimator, prediction, accuracy_knn, accuracy_nb, predict, naive_bayes):
        self.dataset = dataset
        self.dataset_encoded = dataset_encoded
        self.data_train = data_train
        self.data_test = data_test
        self.target_train = target_train
        self.target_test = target_test
        self.knn_estimator = knn_estimator
        self.prediction = prediction
        self.accuracy_knn = accuracy_knn
        self.accuracy_nb = accuracy_nb
        self.predict = predict
        self.naive_bayes = naive_bayes

    ### Encoding Datatset
    def encode_dataset(self, dataset):
        encoder = OrdinalEncoder()
    
        self.dataset_encoded = encoder.fit_transform(self.dataset[['experiences_offered', 'host_name', 'host_since', 'host_location', 'host_response_time', 'host_is_superhost',
                                                        'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'review_scores_value', 'instant_bookable',
                                                        'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'calculated_host_listings_count',
                                                        'reviews_per_month', 'host_response_rate_binned', 'host_verification_binned']])
        return self.dataset_encoded

    encode_dataset(self, dataset)

    ### Split Test Dataset
    def test_dataset(self, dataset_encoded):
      
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(dataset_encoded, dataset['perceived_quality'], test_size=0.2, random_state=42, stratify=dataset['perceived_quality'])

        return self.data_train, self.data_test, self.target_train, self.target_test

    test_dataset(self, dataset_encoded)

    ### Classification Naive Bayes
    def naive_bayes(self, dataset, data_train, target_train, predict, data_test):

        self.naive_bayes = GaussianNB()
        #naive_bayes.fit(data_train, target_train)
        self.naive_bayes.fit(self.dataset_encoded, self.dataset['perceived_quality'])
        self.predict = naive_bayes.predict(self.data_test)

        return self.naive_bayes, self.predict

    naive_bayes(self, dataset, data_train, target_data, predict, data_test)

    ### Evaluation of Naive Bayes
    def evaluation_naive_bayes(target_test, predict):

        self.accuracy_nb = accuracy_score(self.target_test, self.predict)

    ### Classification K_Nearest_Neighbor
    def k_nearest_neighbor(self, dataset, dataset_encoded, prediction):

        self.knn_estimator = KNeighborsClassifier(7)
        self.knn_estimator.fit(self.dataset_encoded, self.dataset['perceived_quality'])
        #self.knn_estimator.fit(self.data_train, self.target_train)
        self.prediction = self.knn_estimator.predict(self.data_test)

        return self.knn_estimator, self.prediction

    k_nearest_neighbor(self, dataset, dataset_encoded, prediction)

    ### Evaluation of K_Nearest_Neighbor
    def evaluation_k_nearest_neighbor(self, knn_estimator, prediction, target_test):

        self.accuracy_knn = self.knn_estimator.score(self.target_test, self.prediction)

        return self.accuracy_knn

    result = evaluation_k_nearest_neighbor(self, knn_estimator, prediction, target_test)
    print(result)
 