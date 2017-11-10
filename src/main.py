import pandas as pd

def inspect_dataset(dataset):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 50)

    # Access column id
    #print(dataset["id"])
    # Join Column names and print them 
    #print('\n'.join(list(dataset)))
    # Example values
    print(dataset.iloc[7])
    
def main():
    print('Listings Subset: ' + '\n')
    listings_subset = pd.read_csv('../data/subset/listings_sub.csv')
    inspect_dataset(listings_subset)

    print(('\n' * 2) + 'Reviews Subset: ' + '\n')
    reviews_subset = pd.read_csv('../data/subset/reviews_sub.csv')
    inspect_dataset(reviews_subset)

if __name__ == '__main__':
    main()