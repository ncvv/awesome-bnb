''' Main entry point of the project, where the program is launched. '''

import sys

import requests

import io_util as io
import preprocess as pp
#import classifier2 as cl

def main(renew_listings=False):
    ''' Main method. '''
    if renew_listings:
        listings = io.read_csv('../data/original/listings.csv')
        pp.Preprocessor.prepare_listings_data(listings)

    reviews = io.read_csv('../data/original/reviews.csv')
    listings = io.read_csv('../data/processed/listings_processed.csv')
    listings_text = io.read_csv('../data/processed/listings_text_processed.csv')
    preprocessor = pp.Preprocessor(False, requests.Session(), listings, listings_text, reviews)
    preprocessor.process()

    # For Tableau visualization by Helene, just ignore.
    #data = io.read_csv('../data/playground/dataset.csv')
    #plot_data = data[['id', 'longitude', 'latitude', 'perceived_quality']]
    #io.write_csv(plot_data, '../data/playground/visualization.csv')

    # Classification
    dataset = io.read_csv('../data/playground/dataset.csv')
    dataset.drop('id', axis=1, inplace=True)
    
    idx_amenity = dataset.columns.get_loc("Amenity_TV")
    idx_amenity_end = dataset.columns.get_loc ("Amenity_Paidparkingoffpremises")
    
    #idx_tfidf_transit = dataset.columns.get_loc("")
    #idx_tfidf_transit_end = dataset.columns.get_loc("")
    
    #idx_tfidf_description = dataset.columns.get_loc("")
    #idx_tfidf_description_end = dataset.columns.get_loc("")
    
    #idx_tfidf_neighborhood = dataset.columns.get_loc("")
    #idx_tfidf_neighborhood_end = dataset.columns.get_loc("")
    
    #idx_tfidf_house_rules = dataset.columns.get_loc("")
    #idx_tfidf_house_rules_end = dataset.columns.get_loc("")
    
    #print(dataset.shape[1])
    #dataset.drop(dataset.columns[idx_amenity-1: idx_amenity_end+1], axis =1, inplace=True)
    #print(dataset.shape[1])
    
    #classifier = cl.Classifier(dataset)
    #for kn in range(1, 10):
        #classifier.classify_knn(dataset, n=kn)
    #classifier.classify_nb()
    #..
    #print(classifier)

if __name__ == '__main__':
    if sys.argv[1:]:
        renew_listings_flag = sys.argv[1]
        main(renew_listings_flag)
    else:
        main()
