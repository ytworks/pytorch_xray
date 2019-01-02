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
* config_release (良い結果が出た設定ファイル置き場)
* experiments (実験的な設定ファイル置き場)
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

  `` bash ./tools/sample_JSRT_download.sh -f ./sample_data ``

2. 画像データの解凍

  `` cd ./sample_data ``
  `` unzip Nodule154images.zip ``
  `` unzip NonNodule93images.zip ``

3. 教師データの作成
4. dicomをPNGファイルに変換する
5. 設定ファイルを準備する
6. 訓練データとテストデータを分割する
7. 訓練を実行する
8. 評価モデルを作成する
9. 予測を実行してみる


# データ作成ツールの使い方
* JSRTの胸部X線のデータダウンロードスクリプトの使い方
  `` bash ./tools/sample_JSRT_download.sh -f (ダウンロードしたい場所を指定する) ``

# 教師データの作成方法
  1. ファイル名
  2. 所見 (複数の場合は | で区切る)
  3. 空欄
  4. 患者ID or dummy
  の形のcsvファイルを作成してください

# NIH chest-xray14の事前学習の実行方法
