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
