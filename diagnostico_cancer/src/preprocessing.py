import pandas as pd

def one_hot_encoder(dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    '''
    '''
    for column in columns:
        unique_values = dataset[column].dropna().unique()
        
        for value in unique_values:
            new_column = f"{column}_{value}"
            dataset[new_column] = (dataset[column] == value).astype(int)
        
        dataset.drop(column, axis=1, inplace=True)
    
    return dataset

def normalize():
    return

def handle_missing_values():
    return
