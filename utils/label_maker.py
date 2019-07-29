import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def get_label(ini):
    df, findings = get_label_list(ini)
    mlb = MultiLabelBinarizer()
    mlb.fit_transform([set(findings)])
    print('Labels: ', list(mlb.classes_))
    labels = {}
    for index, row in df.iterrows():
        labels.setdefault(row['filepath'], {})
        if 'No Finding' in row['findings']:
            labels[row['filepath']].setdefault('findings', 'normal')
            labels[row['filepath']].setdefault(
                'label', np.array(mlb.transform([set(['normal'])])[0]).astype(np.float32))
        else:
            labels[row['filepath']].setdefault('findings', 'abnormal')
            labels[row['filepath']].setdefault(
                'label', np.array(mlb.transform([set(['abnormal'])])[0]).astype(np.float32))
        labels[row['filepath']].setdefault('patient_id', row['patient_id'])
        assert len(labels[row['filepath']]['label']) == ini.getint('network', 'num_classes'), "Binarizer Error"
    return labels, list(mlb.classes_)


def get_label_list(ini):
    df = pd.read_csv(ini.get('data', 'label_path'),
                     usecols=[0, 1, 3])
    df.columns = ['filepath', 'findings', 'patient_id']
    findings = list(set(['normal', 'abnormal']))
    assert len(findings) == ini.getint('network', 'num_classes'), 'Luck of some findings'
    return df, findings
