FROM tleyden5iwx/caffe-gpu-master
MAINTAINER Michael Bartoli <michael.bartoli@pomona.edu>

RUN apt-get update
RUN apt-get -y install \
	python \
	build-essential \
	python-dev \
	python-pip \
	wget \
	unzip \
	cmake \
	ipython \
	git \
	perl \
	libatlas-base-dev \
	gcc \
	gfortran \
	g++ \
	libblas-dev \
	libblas-doc \
	liblapacke-dev \
	liblapack-doc
RUN apt-get install -f
RUN pip install numpy scipy nose sklearn nltk

ENV CUDA_RUN http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

RUN cd /opt && \
  wget $CUDA_RUN && \
  chmod +x *.run && \
  mkdir nvidia_installers && \
  ./cuda_6.5.14_linux_64.run -extract=`pwd`/nvidia_installers && \
  cd nvidia_installers && \
  ./NVIDIA-Linux-x86_64-340.29.run -s -N --no-kernel-module

RUN cd /opt/nvidia_installers && \
  ./cuda-linux64-rel-6.5.14-18749181.run -noprompt

ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-6.5/lib64
ENV PATH=$PATH:/usr/local/cuda-6.5/bin

RUN pip install Theano

WORKDIR /home
RUN git clone https://github.com/mbartoli/deep-simplification
WORKDIR /home/deep-simplification
