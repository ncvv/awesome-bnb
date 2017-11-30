''' Main entry point of the project, where the program is launched. '''

import sys

import requests

import io_util as io
import preprocess as pp
import classifier as cl

def main(renew_listings=False):
    ''' Main method. '''
    long_tfidf = False
    num_labels = 2
    
    if renew_listings:
        listings = io.read_csv('../data/original/listings.csv')
        pp.Preprocessor.prepare_listings_data(listings)
        # Preprocessing
        reviews = io.read_csv('../data/original/reviews.csv')
        listings = io.read_csv('../data/processed/listings_processed.csv')
        listings_text = io.read_csv('../data/processed/listings_text_processed.csv')
        preprocessor = pp.Preprocessor(False, requests.Session(), listings, listings_text, reviews)
        preprocessor.process(num_labels, long_tfidf)

    file_name = 'dataset'
    file_name += '_' + str(num_labels)
    if long_tfidf:
        file_name += '_long_tfidf'

    dataset = io.read_csv('../data/final/' + file_name + '.csv')
    
    print('#Columns: ' + str(len(list(dataset))))
    print('#Rows: ' + str(len(dataset)) + '\n')
    
    encoded_file_path = '../data/final/' + file_name + '_encoded.csv'
    classifier = cl.Classifier(dataset, encoded_file_path, long_tfidf=long_tfidf, display_columns=False, scoring='accuracy')
    
    # Parameter Tuning
    #classifier.para_tuning_dt()
    #classifier.para_tuning_SVM(loose=True, fine=False, use_sample=False)
    #classifier.para_tuning_knn()
    #classifier.para_tuning_nc()

    # Classification
    cross_validate = True
    for kn in range(2, 7):
        classifier.classify_knn(n=kn)
    classifier.classify_knn(n=classifier.accuracy_knn_n, cross_validate=cross_validate, display_matrix=True) # Leave as is, prints the CM and CR for kNN's best n.
    print('-' * 52)
    classifier.classify_nb(cross_validate=cross_validate)
    print('-' * 52)
    classifier.classify_mnb(cross_validate=cross_validate)
    print('-' * 52)
    classifier.classify_svm(cross_validate=cross_validate, display_roc=True)
    print('-' * 52)
    classifier.classify_nc(cross_validate=cross_validate)
    print('-' * 52)
    classifier.classify_dt(cross_validate=cross_validate)
    print('-' * 52)

    print(classifier)
    classifier.plot_roc()

if __name__ == '__main__':
    ''' Usage: ~/src python main.py RENEW_LISTINGS_FLAG\n\n    RENEW_LISTINGS_FLAG determines whether preprocessing and encoding is executed again. '''
    if sys.argv[1:]:
        renew_listings_flag = sys.argv[1]
        main(renew_listings_flag)
    else:
        main()
