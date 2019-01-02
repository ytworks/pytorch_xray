#! /bin/bash

cd `dirname $0`
cd ..

FILE_PATH=""
while getopts hf: OPT
do
    case $OPT in
      h)  usage_exit
          ;;
      f)  FILE_PATH=$OPTARG
          ;;
      \?) usage_exit
          ;;
    esac
done


mkdir $FILE_PATH
cd $FILE_PATH
# Download Chest X-ray DICOM files
wget http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2017/11/Nodule154images.zip
wget http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2016/01/NonNodule93images.zip
