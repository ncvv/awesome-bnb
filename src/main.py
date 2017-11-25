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

    # For Tableau visualization by Helene, just ignore.
    #data = io.read_csv('../data/playground/dataset.csv')
    #plot_data = data[['id', 'longitude', 'latitude', 'perceived_quality']]
    #io.write_csv(plot_data, '../data/playground/visualization.csv')

    # Classification
    dataset = io.read_csv('../data/final/dataset.csv')
    dataset.drop('id', axis=1, inplace=True)
    
    encoded_file_path = '../data/final/encoded_dataset'
    ### Warning: If line 230ff in preprocess.py ("mehr features") are used, .get_loc("") have to be adjusted.
    # Drop Amenities
    #idx_amenity = dataset.columns.get_loc("Amenity_TV")
    #idx_amenity_end = dataset.columns.get_loc ("Amenity_Paidparkingoffpremises") + 1
    #dataset.drop(dataset.columns[idx_amenity: idx_amenity_end], axis=1, inplace=True)
    #encoded_file_path += '_woamenities'

    # Drop Transit TFIDF
    #idx_tfidf_transit = dataset.columns.get_loc("transit_10")
    #idx_tfidf_transit_end = dataset.columns.get_loc("transit_west") + 1
    #dataset.drop(dataset.columns[idx_tfidf_transit: idx_tfidf_transit_end], axis=1, inplace=True)
    #encoded_file_path += '_wotransit'
    
    # Drop Description TFIDF
    #idx_tfidf_description = dataset.columns.get_loc("description_access")
    #idx_tfidf_description_end = dataset.columns.get_loc("description_walk") + 1
    #dataset.drop(dataset.columns[idx_tfidf_description: idx_tfidf_description_end], axis=1, inplace=True)
    #encoded_file_path += '_wodescr'
    
    # Drop Neighborhood TFIDF
    #idx_tfidf_neighborhood = dataset.columns.get_loc("neighborhood_overview_area")
    #idx_tfidf_neighborhood_end = dataset.columns.get_loc("neighborhood_overview_walk") + 1
    #dataset.drop(dataset.columns[idx_tfidf_neighborhood: idx_tfidf_neighborhood_end], axis=1, inplace=True)
    #encoded_file_path += '_woneighbor'
    
    # Drop House Rules TFIDF
    #idx_tfidf_house_rules = dataset.columns.get_loc("house_rules_allow")
    #idx_tfidf_house_rules_end = dataset.columns.get_loc("house_rules_use") + 1
    #dataset.drop(dataset.columns[idx_tfidf_house_rules: idx_tfidf_house_rules_end], axis=1, inplace=True)
    #encoded_file_path += '_wohrules'

    # Print Column Names
    print('Columns:\n' + '\n'.join(list(dataset)) + '\n')
    print('#Columns: ' + str(len(list(dataset))))
    print('#Rows: ' + str(len(dataset)) + '\n')

    # TBD in Testing:
    # 1. unnötige columns vom orig. dataset rausschmeißen
    # 2. andere NB Classifier
    # 3. Compute other features? e.g. Num Amenities
    encoded_file_path += '.csv'
    classifier = cl.Classifier(dataset, encoded_file_path)
    #for kn in range(1, 10):
        #classifier.classify_knn(n=kn)
    classifier.classify_nb()
    classifier.classify_svm()
    #..
    print(classifier)

if __name__ == '__main__':
    if sys.argv[1:]:
        renew_listings_flag = sys.argv[1]
        main(renew_listings_flag)
    else:
        main()
