''' Utility methods. '''

import os
import csv

import pandas as pd

def remove_line_by_id_df(df, value, col_name='id'):
    ''' Remove a line from a dataframe in the given column with a specific value.
        If more than one line apply, remove all of them.
        The current implementation regards the removal of lines e.g. with a single id.
        This can be altered (-> in another method) to remove lines with e.g. price > val. '''
    return df[getattr(df, col_name) != value]

def remove_lines_by_id_df(df, ids):
    ''' Remove a list or set of ids from the dataframe and return it. '''
    if ids:
        for i in ids:
            df = remove_line_by_id_df(df, i)
    return df

def append_dict_as_column_df(df, col_name, dct, key='id'):
    ''' Takes a dictionary and appends it as a column to a dataframe based on key values.
        Example:
        >>> d = {'id': [123, 445, 999], 'email': [0, 0, 0]}
        >>> df = pd.DataFrame(d)

        >>> print(df)
             id     email
        0    123    0
        1    445    0
        2    999    0

        >>> dct = {123: 'abc@gmail.de', 445: 'ABC@mail.uni-mannheim.de', 912: 'ABC@yahoo.de'}

        >>> df['email'] = df['id'].map(dct)

        >>> print(df)
             id     email
        0    123    'abc@gmail.de'
        1    445    'ABC@mail.uni-mannheim.de'
        2    999    NaN                         '''
    df[col_name] = df[key].map(dct)
    return df

def get_column_as_dict_df(df, col_name):
    ''' Return the given column in a dict with key: id, value: value of the column. '''
    return pd.Series(df[col_name].values.tolist(), index=df['id'].values.tolist()).to_dict()

def get_column_as_list_df(df, col_name):
    ''' Return the given column as a list, without pandas index. '''
    return df[col_name].values.tolist()

def merge_df(df, df_m, key='id'):
    ''' Merge two dataframes on the given key. (Suitable for merging listings_processed and listings_text_processed.) '''
    return pd.merge(df, df_m, on=key)

def read_csv(path):
    ''' Read in a .csv file and return it as a pandas df. '''
    return pd.read_csv(get_universal_path(path), encoding='utf-8')

def append_line_csv(path, lst):
    ''' Append a list of one or more lines to an existing csv file. '''
    with open(get_universal_path(path), 'a+', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if not isinstance(lst[0], list):
            writer.writerow(lst)
        else:
            for line in lst:
                writer.writerow(line)

def write_csv(df, path, index=False):
    ''' Write a pandas dataframe to the given path as .csv file. '''
    df.to_csv(get_universal_path(path), sep=',', encoding='utf-8', index=index)

def get_universal_path(file_path):
    ''' Return universal path to file that works on every operating system. '''
    args = file_path.split('/')
    return os.path.join(*args)
