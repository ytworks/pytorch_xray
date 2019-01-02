import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc
import argparse

import csv
import sys


def write_fig(test, prob, figname):
    fpr, tpr, thresholds = roc_curve(test, prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    pl.figure()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    for i in range(0, len(fpr), int(len(fpr) / 5)):
        pl.text(fpr[i], tpr[i], '%0.5f' % thresholds[i], fontsize=8)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC')
    pl.legend(loc="lower right")
    pl.savefig(figname)
    return roc_auc


parser = argparse.ArgumentParser(description='パラメータファイルの読み込み')
parser.add_argument('-csv', required=True)
parser.add_argument('-png', required=True)
args = parser.parse_args()
f = csv.reader(open(args.csv, 'r'), lineterminator='\n')
test, prob = [], []
for row in f:
    test.append(int(float(row[1])))
    prob.append(float(row[0]))
roc = write_fig(test, prob, args.png)
