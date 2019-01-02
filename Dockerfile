# nvidiaからdockerイメージをダウンロードする場合
#FROM nvcr.io/nvidia/pytorch:18.12.1-py3

# 自作で途中まで作ったdockerイメージを利用する場合(今回はgpu:1.0としている)
FROM gpu:1.0

# set timezone
RUN export TZ=Asia/Tokyo

# app env
RUN mkdir /app && \
    mkdir /app/data && \
    mkdir /app/work && \
    mkdir /app/src


# pip install
ADD . /app/src
WORKDIR /app/src

RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev




RUN pip install -r /app/src/requirements.txt
