import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def KNN(dataset, numeric_cols, k=5):
    df_numeric = dataset[numeric_cols].copy()
    df_filled = df_numeric.copy()

    # Escalamos los datos (z-score)
    means = df_numeric.mean()
    stds = df_numeric.std()
    df_scaled = (df_numeric - means) / stds

    # Iteramos columna por columna
    for col in numeric_cols:
        # Filas donde hay NaN en esta columna
        missing_idx = df_scaled[df_scaled[col].isna()].index

        for idx in missing_idx:
            row = df_scaled.loc[idx]

            # Usamos solo las columnas que no tienen NaNs en esta fila (para calcular la distancia)
            available_features = row.dropna().index
            other_rows = df_scaled.loc[df_scaled[col].notna(), available_features]

            # Calculamos distancias euclidianas a otras filas sin NaN en esta columna
            distances = ((other_rows - row[available_features]) ** 2).sum(axis=1).pow(0.5)

            # Tomamos los k vecinos más cercanos
            nearest = distances.nsmallest(k).index

            # Imputamos con el promedio (en la columna original sin escalar)
            imputed_value = df_numeric.loc[nearest, col].mean()
            df_filled.at[idx, col] = imputed_value

    return df_filled

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

def handle_missing_values():
    return

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

def knn_impute_celltype(df, numeric_cols, k=5):
    df_filled = df.copy()

    # Separate known and missing CellType
    known = df[df['CellType'].notna()]
    missing = df[df['CellType'].isna()]

    for idx in missing.index:
        # Row with missing CellType
        row = df.loc[idx, numeric_cols].values.astype(float)

        # Compute Euclidean distances to all known rows
        known_values = known[numeric_cols].values.astype(float)
        distances = np.linalg.norm(known_values - row, axis=1)

        # Get k nearest neighbors
        nearest_indices = distances.argsort()[:k]
        neighbor_celltypes = known.iloc[nearest_indices]['CellType']

        # Compute mode manually
        mode_value = neighbor_celltypes.value_counts().index[0]

        # Fill in the missing value
        df_filled.at[idx, 'CellType'] = mode_value

    return df_filled

def normalize_train_test(train:pd.DataFrame, test:pd.DataFrame, columns:list[str]):
    normalized_train, stats = normalize(train, columns)
    normalized_test, _ = normalize(test, columns, stats)
    return normalized_train, normalized_test