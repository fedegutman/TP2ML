import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm

# EMPROLIJAR TODAS LAS FUNCIONES
    
def categorical_KNN(dataset: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], k=5):
    df = dataset.copy()
    df_numeric = df[numeric_cols]

    means = df_numeric.mean()
    stds = df_numeric.std()
    df_scaled = (df_numeric - means) / stds

    for col in categorical_cols:
        missing_idx = df[df[col].isna()].index

        for index in missing_idx:
            row = df_scaled.loc[index]

            # Candidatos: filas donde col no sea NaN
            candidates = df_scaled.loc[df[col].notna()]
            distances = ((candidates - row) ** 2).sum(axis=1) ** 0.5

            neighbors_idx = distances.nsmallest(k).index
            neighbor_values = df.loc[neighbors_idx, col].dropna()

            if not neighbor_values.empty:
                mode = neighbor_values.value_counts().idxmax()
                df.at[index, col] = mode

    return df

def numeric_KNN(dataset: pd.DataFrame, numeric_cols: list[str], k=5):
    df_numeric = dataset[numeric_cols].copy()
    new_df = df_numeric.copy()

    means = df_numeric.mean()
    stds = df_numeric.std()

    for col in numeric_cols:
        # Guardar los NaNs originales del resto de las columnas
        other_cols = [c for c in numeric_cols if c != col]
        original_nans = df_numeric[other_cols].isna()

        # Reemplazar NaNs en las otras columnas con la media
        temp_df = df_numeric.copy()
        for c in other_cols:
            temp_df[c] = temp_df[c].fillna(means[c])

        df_scaled = (temp_df - means) / stds

        missing_idx = df_numeric[df_numeric[col].isna()].index

        for index in missing_idx:
            row = df_scaled.loc[index]
            features = row[other_cols]

            candidates = df_scaled.loc[df_numeric[col].notna(), other_cols]

            distances = ((candidates - features) ** 2).sum(axis=1) ** 0.5

            neighbors = distances.nsmallest(k).index
            new_value = df_numeric.loc[neighbors, col].mean()
            new_df.at[index, col] = new_value

        for c in other_cols:
            df_numeric.loc[original_nans[c], c] = np.nan

    result = dataset.copy()
    result[numeric_cols] = new_df

    return result
    
def test_categorical_KNN(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], k=5):
    train_df = train_dataset.copy()
    test_df = test_dataset.copy()

    # Escalar los numéricos usando estadísticas del train
    means = train_df[numeric_cols].mean()
    stds = train_df[numeric_cols].std()

    train_scaled = (train_df[numeric_cols] - means) / stds
    test_scaled = (test_df[numeric_cols] - means) / stds  # usar las mismas estadísticas

    for col in categorical_cols:
        missing_idx = test_df[test_df[col].isna()].index

        # Candidatos: train donde col no sea NaN
        print(train_scaled)
        candidates = train_scaled[train_df[col].notna()]
        print(candidates)
        candidate_values = train_df[train_df[col].notna()][col]

        for index in missing_idx:
            row = test_scaled.loc[index]

            # Calcular distancias euclidianas
            distances = ((candidates - row) ** 2).sum(axis=1) ** 0.5

            neighbors_idx = distances.nsmallest(k).index
            neighbor_values = candidate_values.loc[neighbors_idx]

            if not neighbor_values.empty:
                mode = neighbor_values.value_counts().idxmax()
                test_df.at[index, col] = mode

    return test_df

def test_numeric_KNN(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, numeric_cols: list[str], k: int = 5):

    train_df = train_dataset[numeric_cols].copy()
    test_df = test_dataset[numeric_cols].copy()
    new_test_df = test_df.copy()

    means = train_df.mean()
    stds = train_df.std()

    for col in numeric_cols:
        other_cols = [c for c in numeric_cols if c != col]

        # Reemplazar NaNs en otras columnas por la media del TRAIN (en ambos datasets)
        train_temp = train_df[other_cols].fillna(means[other_cols])
        test_temp = test_df[other_cols].fillna(means[other_cols])

        # Escalar con media y std del TRAIN
        train_scaled = (train_temp - means[other_cols]) / stds[other_cols]
        test_scaled = (test_temp - means[other_cols]) / stds[other_cols]

        missing_idx = test_df[test_df[col].isna()].index

        for index in missing_idx:
            row = test_scaled.loc[index]

            # Candidatos: filas del train donde la columna a imputar NO sea NaN
            candidates = train_scaled.loc[train_df[col].notna()]
            candidate_values = train_df.loc[train_df[col].notna(), col]

            # Calcular distancias
            distances = ((candidates - row) ** 2).sum(axis=1) ** 0.5

            neighbors = distances.nsmallest(k).index
            new_value = candidate_values.loc[neighbors].mean()

            new_test_df.at[index, col] = new_value

    # Combinar columnas imputadas con el resto del test
    result = test_dataset.copy()
    result[numeric_cols] = new_test_df

    return result

def corr_matrix(dataset:pd.DataFrame, categorical_columns=list[str]):
    encoded_dataset = dataset.copy()
    encoded_dataset = one_hot_encoder(encoded_dataset, columns=categorical_columns)
    corr_matrix = encoded_dataset.corr()

    plt.figure(figsize=(15, 13))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm") # Grafico la matriz de correlación
    plt.show()

def replace_outliers(dataset:pd.DataFrame, column:str):
    '''
    '''
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = dataset[column].apply(
        lambda x: x < lower_bound or x > upper_bound
    ).sum()
    
    dataset[column] = dataset[column].apply(
        lambda x: np.nan if x < lower_bound or x > upper_bound else x
    )
    
    print(f'Number of outliers replaced in column [{column}]: {outliers_count}')    
    return dataset
    
def one_hot_encoder(dataset:pd.DataFrame, columns:list[str]) -> pd.DataFrame:
    '''
    '''
    for column in columns:
        unique_values = dataset[column].dropna().unique()
        
        for value in unique_values:
            new_column = f"{column}_{value}"
            dataset[new_column] = (dataset[column] == value).astype(int)
        
        dataset = dataset.drop(column, axis=1)
    
    return dataset

def normalize(dataset:pd.DataFrame, columns:list[str], stats=None):
    dataset = dataset.copy()
    if stats:
        for i, col in enumerate(columns):
            min_val, max_val = stats[i]
            dataset[col] = (dataset[col] - min_val)/(max_val - min_val)
    
    else:
        stats = []
        for i, col in enumerate(columns):
            min_val, max_val = dataset[col].max(), dataset[col].min()
            stats.append((min_val, max_val))
            dataset[col] = (dataset[col] - min_val)/(max_val - min_val)
    
    return dataset, stats

def standarize():
    return

def handle_missing_values(dataset: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], k=5):
    df = dataset.copy()
    df = numeric_KNN(df, numeric_cols, k)
    df = categorical_KNN(df, numeric_cols, categorical_cols, k)
    return df

def set_range(dataset:pd.DataFrame, range:dict):
    columns = range.keys()
    ranges = range.values()
    for col, (floor, roof) in zip(columns, ranges):
        dataset.loc[dataset[col] < floor, col] = np.nan
        dataset.loc[dataset[col] > roof, col] = np.nan

    return dataset

def remove_missing_rows(dataset:pd.DataFrame, threshold:float):
    nan_counts = dataset.isnull().sum(axis=1)
    nan_percentage = (nan_counts / dataset.shape[1]) * 100

    filtered_dataset = dataset[nan_percentage <= threshold]
    print(f'Removed {len(dataset) - len(filtered_dataset)} rows with more than {threshold}% NaNs.')
    return filtered_dataset

def replace_missing_values(dataset:pd.DataFrame, columns:list[str], keywords:list[str]):
    '''
    '''
    for col in columns:
        dataset.loc[dataset[col].isin(keywords), col] = np.nan    
    return dataset

def replace_nans(dataset:pd.DataFrame, numeric_cols:list[str]):
    df = dataset.copy()
    for col in numeric_cols:
        mean = df[col].mean()
        df[col].fillna(mean)
    return df

def normalize_train_test(train:pd.DataFrame, test:pd.DataFrame, columns:list[str]):
    normalized_train, stats = normalize(train, columns)
    normalized_test, _ = normalize(test, columns, stats)
    return normalized_train, normalized_test

def replace_unwanted_values(dataset:pd.DataFrame, col_range:dict[str:tuple], cat_columns:list[str], keywords:list[str]):
    for column in list(col_range.keys()):
        dataset = replace_outliers(dataset, column)
    
    dataset = set_range(dataset, col_range)
    dataset = replace_missing_values(dataset, cat_columns, keywords)
    
    return dataset

def handle_missing_test_values(test_df:pd.DataFrame, train_df:pd.DataFrame, numeric_cols:list[str], categorical_cols:list[str], k=5):
    test_df = test_df.copy()
    test_df = test_numeric_KNN(train_df, test_df, numeric_cols, k)
    test_df = test_categorical_KNN(train_df, test_df, numeric_cols, categorical_cols, k)
    return test_df
