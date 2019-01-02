import argparse
import glob
from sklearn.model_selection import KFold
import csv
import os

def main():
    parser = argparse.ArgumentParser(description='K-foldクロスバリデーションのデータ分割ファイルの作成')
    parser.add_argument('-pngs', required=True)
    parser.add_argument('-k', required=True)
    parser.add_argument('-csv', required=True)
    args = parser.parse_args()
    pngs = glob.glob(args.pngs + '/' + '*.png')
    kf = KFold(n_splits=int(args.k), shuffle=True)
    for idx, (learn,test) in enumerate(kf.split(pngs)):
        csv_path = args.csv
        with open(csv_path + '/train' + str(idx) + '.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for p in learn:
                writer.writerow([os.path.basename(pngs[p])])
        with open(csv_path + '/test' + str(idx) + '.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for p in test:
                writer.writerow([os.path.basename(pngs[p])])





if __name__=='__main__':
    main()
