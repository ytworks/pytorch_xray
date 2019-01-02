

python train.py -config ./config/sample_chest_xray14.ini --debug
python evaluate.py -config ./config/sample_chest_xray14.ini --debug
python prediction.py -config ./config/sample_chest_xray14.ini -file ./data/JPCLN001.png -dir ./result
