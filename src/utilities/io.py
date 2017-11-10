''' Utility methods '''

import os

import pandas as pd

def read_csv(path):
    ''' Read in a csv as a pandas dataframe. '''
    return pd.read_csv(get_universal_path(path))

def write_csv(dataframe, path):
    ''' Write a dataframe to a path as csv. '''
    dataframe.to_csv(get_universal_path(path), sep='\t', encoding='utf-8')

def get_universal_path(file_path):
    ''' Return universal path to file that works on every operating system. '''
    args = ['..'] + file_path.split('/')
    return os.path.join(*args)
