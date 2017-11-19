
''' Preprocessing module containing all methods of data cleansing and
    tokenizing, stemming as well as stopword removal. '''

import re
import ast

from langdetect import detect

import io_util as io

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

class Preprocessor(object):
    ''' Preprocesses data with different methods. '''

    def __init__(self, crawl, session, listings_data, listings_text_data, reviews_data):
        self.crawl = crawl
        self.ses = session
        self.listings = listings_data
        self.listings_text = listings_text_data
        self.reviews = reviews_data
        self.removal_ids = []
        self.review_removal_ids = []

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
        bins = [0, 50, 85, 99, 100]
        grp_names = ['Bad', 'Medium', 'Good', 'Very Good']
        df['host_response_rate'] = df['host_response_rate'].apply(lambda x: int(x.replace('%', '')))
        df['host_response_rate_binned'] = pd.cut(df['host_response_rate'], bins, labels=grp_names)
        df.loc[df['host_response_rate'] == 0, 'host_response_rate_binned'] = 'Bad'
        df.drop('host_response_rate', axis=1, inplace=True)
        return df
    
    def bin_host_location(self, df):
        df['host_location'] = df['host_location'].apply(lambda x: 'inlondon' if 'lon' in str(x).lower() else 'notinlondon')
        return df

    def clean_zipcodes(self, df):
        ''' Clean the zipcodes and write to clean_zipcodes.csv '''
        dct = io.get_column_as_dict_df(df, 'zipcode')
        for i, zc in dct.items():
            if zc is not None and zc is not np.nan:
                zc = str(zc).upper() 
                split = re.split(r'\s+', zc.strip().replace('\"', ''))
                if split[0].isalnum():
                    dct[i] = split[0]
                else:
                    dct[i] = np.nan
            else:
                dct[i] = np.nan
        return dct

    def delete_dollar(self, df):
        df[['price', 'security_deposit', 'cleaning_fee', 'extra_people']] = df[['price', 'security_deposit', 'cleaning_fee', 'extra_people']].applymap(lambda x: float(str(x).replace('$', '').split('.')[0].replace(',', '.')))
        return df

    def check_language(self, df):
        ''' Remove English reviews. '''
        dct = io.get_column_as_dict_df(df, 'comments')
        for i, comment in dct.items():
            try:
                if detect(comment) != 'en':
                    self.review_removal_ids.append(i)
            except:
                self.review_removal_ids.append(i)

    def bin_host_verification(self, df):
        # Host Verifications: {'jumio', 'sent_id', 'sesame', 'linkedin', 'kba', 'manual_offline', 'facebook', 'offline_government_id', 'photographer', 'amex', 'email', 'phone', 'sesame_offline', 'weibo', 'manual_online', 'government_id', 'google', 'reviews', 'selfie'}
        # Distribution {4: 10142, 3: 5213, 6: 2313, 5: 5078, 2: 339, 7: 475, 8: 78, 1: 10, 9: 5, 10: 1}
        # Max: 19, Most verification methods for one host: 10, Most ofen: 4
        bins = [0, 3, 4, 10]
        grp_names = ['LowerThan4', 'Equals4', 'MoreThan4']
        df['host_verifications'] = df['host_verifications'].apply(lambda x: int(len(ast.literal_eval(x))))
        df['host_verification_binned'] = pd.cut(df['host_verifications'], bins, labels=grp_names)
        df.drop('host_verifications', axis=1, inplace=True)
        return df
    
    def create_label(self, df, num_labels):
        label_name = 'perceived_quality'
        if num_labels == 2:
            bins = [0, 93, 100]
            grp_names = ['Bad', 'Good']
        elif num_labels == 5:
            bins = [0, 65, 75, 90, 95, 100]
            grp_names = ['Catastrophic', 'Bad', 'Medium', 'Good', 'Very Good']
        else:
            return df
        df[label_name] = pd.cut(df['review_scores_rating'], bins, labels=grp_names)
        df.drop('review_scores_rating', axis=1, inplace=True)
        return df

    def process(self):
        ''' Main preprocessing method where all parts are tied together. '''
        # Crawl Airbnb.com page and check if listings are still available
        if self.crawl:
            self.check_onair()

        # Remove lines from pandas dataframe with empty values in these columns
        self.listings = self.listings.dropna(subset=['host_response_rate', 'review_scores_rating', 'security_deposit', 'cleaning_fee', 'bathrooms', 'bedrooms', 'beds'])

        # Remove all rows where number of reviews > 2
        self.listings = self.listings[self.listings.number_of_reviews > 2]
        
        # Parse zipcodes
        self.listings.loc[:, ['zipcode']].fillna(np.nan, inplace=True)
        dct = self.clean_zipcodes(self.listings)
        self.listings = io.append_dict_as_column_df(self.listings, 'zipcode', dct)
        self.listings = self.listings.dropna(subset=['zipcode'])

        # Bin host rate, host location and host verification
        self.listings = self.bin_host_rate(self.listings)
        self.listings = self.bin_host_location(self.listings)
        self.listings = self.bin_host_verification(self.listings)

        # Parse $values
        self.listings = self.delete_dollar(self.listings)

        # Create and append label
        self.listings = self.create_label(self.listings, 2)

        # Remove reviews that are not english
        #self.reviews = self.reviews.dropna(axis=0, how='any')
        #self.check_language(self.reviews)
        #self.reviews = io.remove_lines_by_id_df(self.reviews, self.review_removal_ids)

        # Remove all of the ids in removal_ids from the listings
        self.listings = io.remove_lines_by_id_df(self.listings, self.removal_ids)

        # Remove all of the ids in reviews_removal_ids from the listings
        self.reviews = io.remove_lines_by_id_df(self.reviews, self.review_removal_ids)

        # After all processing steps are done in listings and listings_text_processed, merge them on key = 'id'
        #self.listings = io.merge_df(self.listings, self.listings_text, 'id')

        # After all processing setps are done in reviews.csv and processed text is grouped by id, merge it with listings
        # Maybe we have to overthink this (for example: do we have columns with the same name in the processed listings_text and reviews?
        #                                  Avoid this by appending _lt or _rev at the new columns' names)
        #self.listings = io.merge_df(self.listings, self.reviews, 'id')

        # After all processing steps are done, write the listings file to the playground (this will be changed to ../data/final/_.csv)
        print('#Examples in the end: ' + str(len(self.listings))) # Printing the number of resulting examples for testing purposes and validation
        io.write_csv(self.listings, '../data/playground/dataset.csv')
        #TODO drop 'id'

def prepare_listings_data(listings):
    ''' Process listings and split into two files,
        one file with id and unprocessed text features and (listings_text_processed)
        one file with id and non-textual features (listings_processed). '''
    listings_text = listings[['id', 'transit', 'house_rules', 'amenities', 'description', 'neighborhood_overview']]

    drop_list = ['listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url']
    drop_list.extend(['host_acceptance_rate', 'neighbourhood', 'neighbourhood_group_cleansed', 'license', 'jurisdiction_names', 'has_availability', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'street', 'city', 'state', 'market', 'smart_location', 'country', 'monthly_price', 'weekly_price', 'calendar_last_scraped', 'requires_license'])
    drop_list.extend(['name', 'summary', 'space', 'host_about', 'access', 'interaction', 'notes']) # text that is dropped
    drop_list.extend(['transit', 'house_rules', 'amenities', 'description', 'neighborhood_overview']) # text that is preserved with listing id in listings_text_processed.csv
    drop_list.extend(['square_feet', 'country_code', 'host_id', 'longitude', 'latitude']) # drop other columns that are not valuable for classification but are left in til the end for understanding the dataset and validating it
    listings.drop(drop_list, axis=1, inplace=True)

    io.write_csv(listings, '../data/processed/listings_processed.csv')
    io.write_csv(listings_text, '../data/processed/listings_text_processed.csv')