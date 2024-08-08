import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, cohen_kappa_score
# import matplotlib.pyplot as plt
# import seaborn as sns

def evaluate_predictions(y_true, y_pred, y_score):
    """
    Evaluate predictions using various metrics.
    
    :param y_true: True labels (slide-level)
    :param y_pred: Predicted labels (based on top 10% threshold)
    :param y_score: Raw similarity scores
    :return: Dictionary of metrics
    """
    # Precision, Recall, F1-Score
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # ROC AUC (one-vs-rest for multiclass)
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Cohen's Kappa
    # kappa = cohen_kappa_score(y_true, y_pred)
    
    return {
        # 'precision': precision,
        # 'recall': recall,
        # 'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        # 'cohen_kappa': kappa
    }

def plot_confusion_matrix(cm, class_names, save_path='./confusion_matrix.png'):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    

# Example usage
# Assume you have these arrays:
# y_true: true labels
# y_pred: predicted labels based on top 10% threshold
# y_score: raw similarity scores

