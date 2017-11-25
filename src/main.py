''' Main entry point of the project, where the program is launched. '''

import sys

import requests

import io_util as io
import preprocess as pp
import classifier as cl

def main(renew_listings=False):
    ''' Main method. '''
    if renew_listings:
        listings = io.read_csv('../data/original/listings.csv')
        pp.Preprocessor.prepare_listings_data(listings)
        # Preprocessing
        reviews = io.read_csv('../data/original/reviews.csv')
        listings = io.read_csv('../data/processed/listings_processed.csv')
        listings_text = io.read_csv('../data/processed/listings_text_processed.csv')
        preprocessor = pp.Preprocessor(False, requests.Session(), listings, listings_text, reviews)
        preprocessor.process()

    # Classification
    file_name = 'dataset'
    
    long_tfidf = False
    num_labels = 2

    file_name += '_' + str(num_labels)

    if long_tfidf:
        file_name += '_long_tfidf'
    dataset = io.read_csv('../data/final/' + file_name + '.csv')
    
    encoded_file_path = '../data/final/' + file_name + '_encoded.csv'
    
    print('#Columns: ' + str(len(list(dataset))))
    print('#Rows: ' + str(len(dataset)) + '\n')
    
    classifier = cl.Classifier(dataset, encoded_file_path, long_tfidf=long_tfidf, display_columns=False)
    for kn in range(2, 7):
        classifier.classify_knn(n=kn)
    classifier.classify_nb()
    classifier.classify_svm()
    classifier.classify_nc()
    classifier.classify_dt()
    #..
    print(classifier)

if __name__ == '__main__':
    if sys.argv[1:]:
        renew_listings_flag = sys.argv[1]
        main(renew_listings_flag)
    else:
        main()
