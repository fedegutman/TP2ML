import numpy as np
import pandas as pd
from diagnostico_cancer.src.models import BinaryClassifier
from diagnostico_cancer.src.metrics import extract_results, fscore
from tqdm import tqdm

def train_val_split(dataset:pd.DataFrame, seed, train_size=0.8):
    '''
    Splits a dataset in train and validation sets.

    params:
        df : The input DataFrame containing the data to split. 

    returns:
        train (pd.DataFrame) : The train dataset
        validation (pd.DataFrame) : The validation dataset.
    '''

    train = dataset.sample(frac=train_size,random_state=seed) # ver de hacer con el otro split
    validation = dataset.drop(train.index)
    return train, validation

def cross_val(dataset:pd.DataFrame, target_name:str, lambda_values:list[int], k=int):
    '''cross-validation using f-score as performance metric'''
    df = dataset.copy()
    folds = np.array_split(df, k)

    lambda_scores = {}
    for lmbda in lambda_values:
        lambda_scores[lmbda] = []

    for i in tqdm(range(k), desc="Cross-validation folds"):
        val_set = folds[i]
        train_set = pd.DataFrame()
        
        for j in range(k):
            if j != i:
                train_set = pd.concat([train_set, folds[j]], ignore_index=True)

        for lmbda in tqdm(lambda_values, desc=f"Processing fold {i+1}/{k}", leave=False): #orig: for lmbda in lambda_values
            model = BinaryClassifier(train_set, target_name=target_name, ridge_lambda=lmbda, fit=True)

            val_features = val_set.drop(columns=[target_name]).values
            val_labels = val_set[target_name].values
            val_predictions = model.predict(val_features)

            results = pd.DataFrame({
                target_name: val_labels, 
                'Prediction': val_predictions
            })

            model_results = extract_results(results)
            f_score = fscore(model_results)
            lambda_scores[lmbda].append(f_score)

    fscore_df = pd.DataFrame(lambda_scores, index=range(1, len(lambda_scores[lambda_values[0]]) + 1))    # emprolijar esto
    print('F-scores for each fold and lambda value:')
    print(fscore_df)

    avg_f_scores = {lmbda: np.mean(scores) for lmbda, scores in lambda_scores.items()} # emprolijar esto
    print('Average F-scores for each lambda value')
    for key, mean in avg_f_scores.items():
        print(f"Lambda {key}: F-score = {mean:.4f}")

    return avg_f_scores