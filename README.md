# 画像診断ディープラーニング開発パッケージ

# はじめに
* 画像を投入して画像分類を行うパッケージです
* 複数のアルゴリズムを設定ファイルの変更で実験ができ、出力としては分類結果とClass activation mappingをサポートしています
* 参考: class activation mapping (http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)

# ファイル構成
* train.py (訓練用スクリプト)
* evaluate.py (モデル評価スクリプト)
* prediction.py (予測用スクリプト)
* inference.py (モデル詳細記述スクリプト)
* dataloader.py (データバッチ作成スクリプト)
* utils (各種ライブラリ置き場)
* model (作成したモデル置き場)
* data (データ置き場)　=> https://nihcc.app.box.com/v/ChestXray-NIHCC のメタデータのみ収録済
* config (サンプルの設定ファイル置き場)
* tools (各種データ作成支援ツール)
  * sample_JSRT_download.sh (JSRT胸部X線データ{http://imgcom.jsrt.or.jp/download/}のダウンロードスクリプト)
  * write_roc.py (ROC曲線描画スクリプト)
* README (説明書)
* run.sh (デバック用スクリプト)
* requirements.txt (python環境設定スクリプト)
* Dockerfile (Docker設定ファイル)
* .gitignore (git設定ファイル)

# チュートリアル
1. JSRTの胸部X線データのダウンロード (wgetコマンドが必要なためない場合はインストールしてください)
  * ```bash ./tools/sample_JSRT_download.sh -f ./sample_data```

2. 画像データの解凍
  * ```cd ./sample_data```
  * ```unzip Nodule154images.zip```
  * ```unzip NonNodule93images.zip```
  * ```mkdir dicoms```
  * ```mv ./No*/*.dcm ./dicoms```

3. dicomをPNGファイルに変換する
  * ```mkdir ./pngs```
  * ```cd ..```
  * ```python ./tools/dicom2png.py -dicoms ./sample_data/dicoms -pngs ./sample_data/pngs```


4. 教師データの作成
  * ```python ./tools/make_label.py -pngs ./sample_data/pngs -csv ./sample_data/label.csv```

5. 訓練データとテストデータを分割する
  * ```python ./tools/k_fold_split.py -pngs ./sample_data/pngs -csv ./sample_data -k 3```

6. 設定ファイルを準備する
7. 訓練を実行する
8. 評価モデルを作成する
9. ROCカーブを描画する
10. 予測を実行してみる


# データ作成ツールの使い方
* JSRTの胸部X線のデータダウンロードスクリプトの使い方
  * ```bash ./tools/sample_JSRT_download.sh -f (ダウンロードしたい場所を指定する)````
* JSRT用ラベル作成ツールの使い方
  * ```python ./tools/make_label.py -pngs (pngファイルのディレクトリ) -csv (csvの出力場所)```

* dicom2png変換ツールの使い方
  * ```python ./tools/dicom2png.py -dicoms (dicomファイルのディレクトリ) -pngs (pngの出力先)```

* データ分割ツールの使い方 (k-fold cross validationに対応)
  * ```python ./tools/k_fold_split.py -pngs (pngファイルのディレクトリ) -csv (csvの出力先) -k (k-foldのkの設定)```

* roc描画作成ツールの使い方
  *

# 教師データの作成方法
  1. ファイル名
  2. 所見 (複数の場合は | で区切る)
  3. 空欄
  4. 患者ID or dummy
  の形のcsvファイルを作成してください


# Docker の利用方法
* Docker imageをビルドする
  * ``` docker build -t (作成イメージ名) ./```
* Docker コンテナを起動する
  * ```nvidia-docker run -it --rm -v $PWD/Data:/app/src/Data -v $PWD/Log:/app/src/Log -v $PWD/Pics:/app/src/Pics -v```


# NIH chest-xray14の事前学習の実行方法
  1. https://nihcc.app.box.com/v/ChestXray-NIHCC からデータをダウンロードしてください
  2. チュートリアルと同様の方法で、設定ファイルを準備してください (sample_chest_xray14.iniを編集することで動作します)
