import numpy as np
from sklearn.metrics import roc_auc_score


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
    aucs = []
    for i, (label, pred) in enumerate(zip(labels, preds)):
        try:
            auc = roc_auc_score(label, pred)
        except ValueError:
            print('[Warning] label {} has only one class.'.format(i))
            print(label)
            print(label.shape)
            print(pred)
            auc = 0.5
        aucs.append(auc)

    return aucs
