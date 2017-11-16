
''' Preprocessing module containing all methods of data cleansing and
    tokenizing, stemming as well as stopword removal. '''

import io_util as io

import pandas as pd
import numpy as np

class Preprocessor(object):
    ''' Preprocesses data with different methods. '''

    def __init__(self, session, listings_data, listings_text_data, reviews_data):
        self.crawl = False
        self.ses = session
        self.listings = listings_data
        self.listings_text = listings_text_data
        self.reviews = reviews_data
        self.removal_ids = [15896822] # Test value so list is not empty

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

    def bin_host_rate(self, df):
        ''' Bin the values of host_response_rate (equal width/frequency or even binary). '''
        bins = [0, 50, 95, 100]
        grp_names = ['Bad', 'Medium', 'Good']
        pd.options.mode.chained_assignment = None
        df['host_response_rate'] = df['host_response_rate'].apply(lambda x: int(x.replace('%', '')))
        df['hrr_bins'] = pd.cut(df['host_response_rate'], bins, labels=grp_names)
        df.loc[df['host_response_rate'] == 0, 'hrr_bins'] = 'Bad'
        return df

    def Change_into_Bin_host_location(self):
     with open('../data/processed/listings_processed.csv', 'r') as infile, open('../data/playground/neu1.csv', 'w') as outfile:
       reader = DictReader(infile)
       writer = DictWriter(outfile, fieldnames=reader.fieldnames)
       writer.writeheader()
        for row in reader:
            if "lon" in row['host_location'].lower():
                row['host_location'] = 1
            else:
                row['host_location'] = 0
            writer.writerow(row)  


    def delete_dollar(self):
    with open('listings_processed.csv','r') as infile, open('listings_processed2.csv','w') as outfile:
        reader=csv.DictReader(infile)
        writer=csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for line in reader:
            line['price'] = line['price'].replace('$', '')
            writer.writerow(line)   



    def process(self):
        ''' Main preprocessing method where all parts are tied together. '''
        # Crawl Airbnb.com page and check if listings are still available
        if self.crawl:
            self.check_onair()

        # Remove lines from pandas dataframe with empty values in column host_response_rate
        self.listings = self.listings.dropna(subset = ['host_response_rate'])

        # Bin the host response rate
        self.listings = self.bin_host_rate(self.listings)


        # Remove all of the ids in removal_ids from the listings
        self.listings = io.remove_lines_by_id_df(self.listings, self.removal_ids)

        # After all processing steps are done in listings and listings_text_processed, merge them on key = 'id'
        #self.listings = io.merge_df(self.listings, self.listings_text, 'id')

        # After all processing setps are done in reviews.csv and processed text is grouped by id, merge it with listings
        # Maybe we have to overthink this (for example: do we have columns with the same name in the processed listings_text and reviews?
        #                                  Avoid this by appending _lt or _rev at the new columns' names)
        #self.listings = io.merge_df(self.listings, self.reviews, 'id')

        # After all processing steps are done, write the listings file to the playground (this will be changed to ../data/final/_.csv)
        io.write_csv(self.listings, '../data/playground/dataset.csv')

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
    
    # Remove lines where value for every column is NaN
    #listings = listings.dropna(axis=0, how='all')
    #listings_text = listings_text.dropna(axis=0, how='all')

    # Print type of nan
    #l = io.get_column_as_list_df(listings, 'host_response_rate')
    #for i in l:
    #    if type(i) is not str:
    #        print(str(type(i)) + ': ' + str(i))

    io.write_csv(listings, '../data/processed/listings_processed.csv')
    io.write_csv(listings_text, '../data/processed/listings_text_processed.csv')
