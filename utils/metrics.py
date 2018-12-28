import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def calc_auc(labels, preds):
    """
    calculate AUC score.
    :param labels: np.array, shape=(N, n_class)
    :param preds: np.array, shape=(N, n_class)
    :return aucs: list of float, auc scores for each class
    """
    # auc of each diseases
    labels = labels.T  # (n_class, N)
    preds = preds.T  # (n_class, N)
    aucs, rocs = [], []
    for i, (label, pred) in enumerate(zip(labels, preds)):
        try:
            auc = roc_auc_score(label, pred)
            fpr, tpr, thresholds = roc_curve(label, pred, pos_label=1)
            roc = [[fpr[j], tpr[j], thresholds[j]] for j in range(len(fpr))]
            roc = np.array(roc)
        except ValueError:
            print('[Warning] label {} has only one class.'.format(i))
            print(label)
            print(label.shape)
            print(pred)
            auc = 0.5
            roc = []
        aucs.append(auc)
        rocs.append(roc)

    return aucs, np.array(rocs)
