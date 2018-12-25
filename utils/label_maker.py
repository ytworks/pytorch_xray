import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def get_label(ini):
    df, findings = get_label_list(ini)
    mlb = MultiLabelBinarizer()
    mlb.fit_transform([set(findings)])
    print('Labels: ', list(mlb.classes_))
    assert mlb.transform([set(['Atelectasis'])])[0][0] == 1, 'Binarizer Error'
    labels = {}
    for index, row in df.iterrows():
        labels.setdefault(row['filepath'], {})
        labels[row['filepath']].setdefault('findings', row['findings'])
        labels[row['filepath']].setdefault(
            'label', np.array(mlb.transform([set(row['findings'].split('|'))])[0]).astype(np.float32))
        labels[row['filepath']].setdefault('patient_id', row['patient_id'])
        assert len(labels[row['filepath']]['label']) == 15, "Binarizer Error"
    return labels, list(mlb.classes_)


def get_label_list(ini):
    df = pd.read_csv(ini.get('data', 'label_path'),
                     usecols=[0, 1, 3])
    df.columns = ['filepath', 'findings', 'patient_id']
    findings = list(set([ele for f in df['findings'].unique()
                         for ele in f.split('|')]))
    assert len(findings) == 15, 'Luck of some findings. Expects 15 types'
    return df, findings
