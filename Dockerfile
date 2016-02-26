FROM ubuntu:14.04
MAINTAINER Mike Bartoli <michael.bartoli@pomona.edu>

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose \
  python-numpy \
  python-scipy \
  wget

RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip install ipdb

WORKDIR /home
RUN git clone http://github.com/mbartoli/deep-simplification

WORKDIR /home/deep-simplification/simplify
#RUN wget http:/X/model_hal.iter72000.npz
#RUN wget http://X/model_hal.iter72000.npz.pkl
