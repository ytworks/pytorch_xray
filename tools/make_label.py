import glob
import os
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(description='JSRTの教師データCSVの作成')
    parser.add_argument('-pngs', required=True)
    parser.add_argument('-csv', required=True)
    args = parser.parse_args()
    pngs = glob.glob(args.pngs + '/' + '*.png')
    csv_path = args.csv
    with open(csv_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["Image Index","Finding Labels","Follow-up #","Patient ID"])
        for png in pngs:
            file_name = os.path.basename(png)
            if file_name.find('JPCLN') > -1:
                writer.writerow([file_name, 'abnormal', '', ''])
            else:
                writer.writerow([file_name, 'normal', '', ''])




if __name__=='__main__':
    main()
