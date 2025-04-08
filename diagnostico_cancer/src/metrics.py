import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diagnostico_cancer.src.models import BinaryClassifier
from tqdm import tqdm

# – Curva Precision-Recall (PR)
# Curva ROC
# – AUC-ROC
# – AUC-PR

def cross_val_threshold(dataset:pd.DataFrame, target_name:str, threshold_values:list[float], k=int, lmbda=0): # poner esto en el otro archivo
    '''cross-validation to try different threshold values'''
    df = dataset.copy()
    folds = np.array_split(df, k)

    threshold_scores = {}
    for threshold in threshold_values:
        threshold_scores[threshold] = []

    for i in tqdm(range(k), desc="Cross-validation folds"):
        val_set = folds[i]
        train_set = pd.DataFrame()
        
        for j in range(k):
            if j != i:
                train_set = pd.concat([train_set, folds[j]], ignore_index=True)

        for threshold in tqdm(threshold_values, desc=f"Processing fold {i+1}/{k}", leave=False): #orig: for lmbda in lambda_values
            model = BinaryClassifier(train_set, target_name=target_name, ridge_lambda=lmbda, threshold=threshold, fit=True)

            val_features = val_set.drop(columns=[target_name]).values
            val_labels = val_set[target_name].values
            val_predictions = model.predict(val_features)

            results = pd.DataFrame({
                target_name: val_labels, 
                'Prediction': val_predictions
            })

            model_results = extract_results(results)
            f_score = fscore(model_results)
            threshold_scores[threshold].append(f_score)

    fscore_df = pd.DataFrame(threshold_scores, index=range(1, len(threshold_scores[threshold_values[0]]) + 1))    # emprolijar esto
    print('F-scores for each fold and threshold:')
    print(fscore_df)

    avg_f_scores = {lmbda: np.mean(scores) for lmbda, scores in threshold_scores.items()} # emprolijar esto
    print('Average F-scores for each threshold')
    for key, mean in avg_f_scores.items():
        print(f"Threshold {key:.2f}: F-score = {mean:.4f}")

    return avg_f_scores

def extract_results(results:pd.DataFrame):
    y, y_pred = results.columns
    
    ground_truth = results[y]
    predicted = results[y_pred]

    TP = ((ground_truth == 1) & (predicted == 1)).sum()
    FP = ((ground_truth == 0) & (predicted == 1)).sum()
    TN = ((ground_truth == 0) & (predicted == 0)).sum()
    FN = ((ground_truth == 1) & (predicted == 0)).sum()

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

def accuracy(results:dict[str:int]):
    acc = (results['TP'] + results['TN']) / (results['TP'] + results['FP'] + results['TN'] + results['FN']) 
    return acc

def precision(results:dict[str:int]):
    prec = 0
    if (results['TP'] + results['FP']) != 0:
        prec = results['TP'] / (results['TP'] + results['FP'])
    return prec

def recall(results:dict[str:int]):
    rec = 0
    if (results['TP'] + results['FN']) != 0:
        rec = results['TP'] / (results['TP'] + results['FN'])
    return rec

def confusion_matrix(results:dict[str:int]):
    matrix = [
        [results['TN'], results['FP']],  # Row 1: True Negatives, False Positives
        [results['FN'], results['TP']]   # Row 2: False Negatives, True Positives
    ]

    labels: list[str] = ['Negative', 'Positive']

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def fscore(results:dict[str:int]):
    score = (2*results['TP'])/(2*results['TP'] + results['FP'] + results['FN'])
    return score

def true_positive_rate(results:dict[str:int]):
    tp_rate = results['TP']/(results['TP'] + results['FN'])
    return tp_rate

def false_positive_rate(results:dict[str:int]):
    fp_rate = results['FP']/(results['FP'] + results['TN'])
    return fp_rate

def precision_recall_curve(threshold_metrics:dict[float:dict[str:int]]):
    prec = []
    rec = []

    for threshold in threshold_metrics.keys():
        prec.append(precision(threshold_metrics[threshold]))
        rec.append(recall(threshold_metrics[threshold]))
    plt.plot(rec, prec)
    plt.show()
    
    return prec, rec

def roc_curve(threshold_metrics:dict[float:dict[str:int]]):
    tp_rate = []
    fp_rate = []

    for threshold in threshold_metrics.keys():
        tp_rate.append(true_positive_rate(threshold_metrics[threshold]))
        fp_rate.append(false_positive_rate(threshold_metrics[threshold]))
    plt.plot(fp_rate, tp_rate)
    plt.show()

    return tp_rate, fp_rate

def auc_roc(tp_rate, fp_rate):
    auc = np.trapz(fp_rate, tp_rate)
    return auc

def auc_pr(prec, rec):
    auc = np.trapz(rec, prec)
    return auc

def get_thresholds_results(threshold_values:list[float], train_df, valid_df, target_name):
    threshold_metrics = {}
    model = BinaryClassifier(train_df, target_name=target_name)
    for threshold in threshold_values:
        model.change_threshold(threshold)

        valid_features = valid_df.drop(columns=[target_name]).values
        valid_predictions = model.predict(valid_features)

        results = pd.DataFrame({
            target_name: valid_df[target_name],
            'Prediction': valid_predictions
        })

        threshold_metrics[threshold] = extract_results(results) 

    return threshold_metrics