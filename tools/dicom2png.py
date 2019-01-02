import pydicom
import glob
import os
import argparse
import cv2
import numpy as np

def read_dicom(file_path):
    plan = pydicom.read_file(file_path)
    image_2d = np.array(plan.pixel_array)
    phototype = plan.PhotometricInterpretation
    Bits = plan.WindowWidth
    if phototype == 'MONOCHROME1':
        image_2d = float(Bits - 1) - image_2d
    image = (255 * np.array(image_2d).astype(np.float) /
             np.float(Bits)).astype(np.uint8)
    return np.reshape(image, (image.shape[0], image.shape[1], 1))


def main():
    parser = argparse.ArgumentParser(description='DICOMをPNGへ変換')
    parser.add_argument('-dicoms', required=True)
    parser.add_argument('-pngs', required=True)
    args = parser.parse_args()
    dicoms = args.dicoms
    pngs = args.pngs
    dcm_files = glob.glob(dicoms + '/*.dcm')
    for file_path in dcm_files:
        root, ext = os.path.splitext(os.path.basename(file_path))
        img = read_dicom(file_path)
        print(root)
        cv2.imwrite(pngs + '/' + root +'.png', img)


if __name__=='__main__':
    main()
