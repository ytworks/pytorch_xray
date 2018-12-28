import numpy as np


def sampling(ini, file_list, label_dict):
    label_num = []
    for f, v in label_dict.items():
        label_num.append(v['label'])
    label_num = np.sum(np.array(label_num), axis=0)
    max_num, max_arg = np.max(label_num), np.argmax(label_num)
    new_list = []
    for f in file_list:
        ratio = 0
        for idx, v in enumerate(label_dict[f]['label']):
            if v == 1 and int(max_num / label_num[idx]) > ratio:
                ratio = int(max_num / label_num[idx])
        for n in range(ratio):
            new_list.append(f)
    print('Before file num:', len(file_list), 'Balanced:', len(new_list))
    return new_list
