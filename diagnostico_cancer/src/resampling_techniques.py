import numpy as np
import pandas as pd

def undersampling(df:pd.DataFrame, target_name:str) -> pd.DataFrame:
    dataset = df.copy()
    class_values = dataset[target_name].value_counts()
    bigger_class = class_values.idxmax()
    smaller_class_size = class_values.min()

    bigger_class_data = dataset[dataset[target_name] == bigger_class]
    smaller_class_data = dataset[dataset[target_name] != bigger_class]
    
    bigger_class_batch = bigger_class_data.sample(n=smaller_class_size, random_state=42)
    undersampled_dataset = pd.concat([bigger_class_batch, smaller_class_data])
    
    undersampled_dataset = undersampled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    return undersampled_dataset

def duplication_oversampling(df:pd.DataFrame, target_name:str) -> pd.DataFrame:
    dataset = df.copy()
    class_values = dataset[target_name].value_counts()
    minority_class = class_values.idxmin()

    minority_df = dataset[dataset[target_name] == minority_class]
    majority_df = dataset[dataset[target_name] != minority_class]

    n_duplication = len(majority_df) - len(minority_df)
    duplicated_minority = minority_df.sample(n=n_duplication, replace=True, random_state=42)

    balanced_dataset = pd.concat([dataset, duplicated_minority], ignore_index=True).sample(frac=1, random_state=42)
    return balanced_dataset

def smote_oversampling(dataset:pd.DataFrame):
    return

def cost_reweighting(dataset:pd.DataFrame):
    return

