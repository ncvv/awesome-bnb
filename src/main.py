''' Main entry point of the project, where the program is launched. '''

import requests

import io_util as io
import preprocess as pp

import pandas as pd

def inspect_dataset(dataset):
    ''' Dataset inspection method for getting insights on different features, value examples, .. '''
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 120)

    # Access column id
    print(dataset["id"])
    # Join Column names and print them
    print('\n'.join(list(dataset)))
    # Example values of ex. with id 10
    print(dataset.iloc[10])

def main():
    ''' Main method. '''
    listings = io.read_csv('../data/original/listings.csv')
    pp.prepare_listings_data(listings)

    session = requests.Session()
    reviews = io.read_csv('../data/original/reviews.csv')
    listings = io.read_csv('../data/processed/listings_processed.csv')
    listings_text = io.read_csv('../data/processed/listings_text_processed.csv')
    preprocessor = pp.Preprocessor(False, session, listings, listings_text, reviews)
    preprocessor.process()

if __name__ == '__main__':
    main()
