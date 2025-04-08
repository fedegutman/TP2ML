import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# – Curva Precision-Recall (PR)
# Curva ROC
# – AUC-ROC
# – AUC-PR

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
    prec = results['TP'] / (results['TP'] + results['FP'])
    return prec

def recall(results:dict[str:int]):
    prec = results['TP'] / (results['TP'] + results['FN'])
    return prec

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

def precision_recall(results:dict[str:int]):
    return

def auc_roc_curve(results:dict[str:int]):
    return

def auc_pr_curve(results:dict[str:int]):
    return