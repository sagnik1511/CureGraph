from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, f1_score, recall_score, roc_auc_score)




def calculate_scores(y_true, y_pred, dataset="training"):
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy Score : {round(acc, 4)}")
    prc = precision_score(y_true, y_pred)
    print(f"Precision Score : {round(prc, 4)}")
    rcs = recall_score(y_true, y_pred)
    print(f"recall Score: {round(rcs, 4)}")
    f1s = f1_score(y_true, y_pred)
    print(f"F1 Score: {round(f1s, 4)}")
    ras = roc_auc_score(y_true, y_pred)
    print(f"ROC_AUC Score: {round(ras, 4)}")
    print(confusion_matrix(y_true, y_pred))

    return {
        f"{dataset}_accuracy": acc,
        f"{dataset}_precision": prc,
        f"{dataset}_recall": rcs,
        f"{dataset}_f1_score": f1s,
        f"{dataset}_roc_auc_score": ras
    }

