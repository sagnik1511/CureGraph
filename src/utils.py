import mlflow
from sklearn.metrics import (accuracy_score, precision_score, f1_score,
                              recall_score, roc_auc_score)

def calculate_scores(y_true, y_pred, epoch, dataset="training"):
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy Score : {round(acc, 4)}")
    mlflow.log_metric(f"{dataset}_accuracy", acc, step=epoch)
    prc = precision_score(y_true, y_pred)
    print(f"Precision Score : {round(prc, 4)}")
    mlflow.log_metric(f"{dataset}_precision", acc, step=epoch)
    rcs = recall_score(y_true, y_pred)
    print(f"recall Score: {round(rcs, 4)}")
    mlflow.log_metric(f"{dataset}_recall", acc, step=epoch)
    f1s = f1_score(y_true, y_pred)
    print(f"F1 Score: {round(f1s, 4)}")
    mlflow.log_metric(f"{dataset}_f1_score", acc, step=epoch)
    ras = roc_auc_score(y_true, y_pred)
    print(f"ROC_AUC Score: {round(ras, 4)}")
    mlflow.log_metric(f"{dataset}_roc_auc_score", acc, step=epoch)



