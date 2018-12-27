from sklearn.model_selection import train_test_split
import pandas as pd


def data_split(ini, labels, debug_mode):
    # TBD patient-wise
    tv_list = [x[0] for x in pd.read_csv(
        ini.get('data', 'official_train'), usecols=[0]).values]
    test_list = [x[0] for x in pd.read_csv(
        ini.get('data', 'official_test'), usecols=[0]).values]
    val_ratio = ini.getfloat('params', 'validation_ratio')
    train_ratio = ini.getfloat('params', 'train_ratio')
    test_ratio = ini.getfloat('params', 'test_ratio')
    ratio = val_ratio / (val_ratio + train_ratio)
    train_list, val_list = train_test_split(
        tv_list, test_size=ratio, random_state=None)
    assert len(train_list) + len(val_list) == len(tv_list), "Split Error"
    if debug_mode:
        train_list = train_list[0:300]
        val_list = val_list[0:300]
        test_list = test_list[0:300]
    print("train, validation, test = ", len(
        train_list), len(val_list), len(test_list))
    return train_list, val_list, test_list
