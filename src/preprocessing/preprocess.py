
''' Preprocessing module containing all methods of data cleansing and
    tokenizing,stemming as well as stopword removal. '''

import sys
sys.path.append('../')

import tokenize as tkn
import utilities.io as io

class Preprocessor(object):
    ''' Preprocesses data with different methods. '''

    def __init__(self, session, listings_data, listings_text_data, reviews_data):
        self.crawl = False
        self.ses = session
        self.listings = listings_data
        self.listings_text = listings_text_data
        self.reviews = reviews_data

    # See commit 8f9d8e1f for implementation of writing results to a .csv.
    def check_onair(self):
        ''' Checks how many of the listing ids are still online@Airbnb. '''
        air_url = 'https://www.airbnb.de/rooms/'
        ids = self.listings['id']
        num_apts = len(ids)
        for i in ids.tolist():
            url = air_url + str(i)
            try:
                # content-length only in header if listing is not available anymore (not on Air)
                content_length = self.ses.get(url).headers.__getitem__('content-length')
                print(str(i) + ' is not on Air anymore; content-length: ' + str(content_length))
                num_apts -= 1
            except KeyError:
                print(str(i) + ' is still on Air.')
        print('{0:.2f}% of Apartments are still on Air.'.format(float(num_apts / len(ids)) * 100))
    
    ### Insert methods here
    #def example_method(self, further_parameters, default_parameter=5):
    #    print('this is a ' + str(further_parameters) + ' with default_parameter=' + str(default_parameter))

    ###

    def process(self):
        ''' Main preprocessing method where all parts are tied together. '''
        # Crawl Airbnb.com page and check if listings are still available
        if self.crawl:
            self.check_onair()
        # Remove lines from pandas dataframe with empty values in columns id, host_id and square_feet
        io.remove_empty_lines_df(self.listings, ['id', 'host_id', 'square_feet'])
        
        ### Insert method calls here
        #self.example_method('test')
        #self.example_method(further_parameters='test') #same result
        #self.example_method(further_parameters='test', 4) #different default parameter
        #self.example_method(further_parameters='test', default_parameter=4) #same result again

        
        ###

        # After all processing steps are done in listings and listings_text_processed, merge them on key = 'id'
        self.listings = io.merge_df(self.listings, self.listings_text, 'id')

        # After all processing setps are done in reviews.csv and processed text is grouped by id, merge it with listings
        # Maybe we have to overthink this (for example: do we have columns with the same name in the processed listings_text and reviews?
        #                                  Avoid this by appending _lt or _rev at the new columns' names)
        self.listings = io.merge_df(self.listings, self.reviews, 'id')
        
        # After all processing steps are done, write the listings file to the playground (this will be changed to ../data/final/_.csv)
        #io.write_csv(self.listings, '../data/playground/dataset.csv')

def process_listings(listings):
    ''' Process listings and split into two files,
        one file with id and unprocessed text features and (listings_text_processed)
        one file with id and non-textual features (listings_processed). '''
    listings_text = listings[['id', 'transit', 'house_rules', 'amenities', 'description', 'neighborhood_overview']]

    drop_list = ['listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url']
    drop_list.extend(['host_acceptance_rate', 'neighbourhood', 'neighbourhood_group_cleansed', 'license', 'jurisdiction_names', 'has_availability', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'street', 'city', 'state', 'market', 'smart_location', 'country', 'monthly_price', 'weekly_price', 'calendar_last_scraped', 'requires_license'])
    drop_list.extend(['name', 'summary', 'space', 'host_about', 'access', 'interaction', 'notes']) # text that is dropped
    drop_list.extend(['transit', 'house_rules', 'amenities', 'description', 'neighborhood_overview']) # text that is preserved with listing id in listings_text_processed.csv
    listings.drop(drop_list, axis=1, inplace=True)

    io.write_csv(listings, '../data/processed/listings_processed.csv')
    io.write_csv(listings_text, '../data/processed/listings_text_processed.csv')
