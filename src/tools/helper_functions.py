def unique_values( table):
    for col in table:
        print(col)
        print(table[col].unique())
        print("\n")
