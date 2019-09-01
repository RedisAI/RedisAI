ARG OS=debian:buster

#----------------------------------------------------------------------------------------------
# FROM insready/bazel:latest as bazel
FROM ${OS}

ARG TF_VER=v1.13.1

ENV X_NPROC "cat /proc/cpuinfo|grep processor|wc -l"

RUN set -ex; apt-get update; apt-get install -y git

WORKDIR /build

# COPY --from=bazel /usr/bin/bazel* /usr/local/bin/
COPY ../../deps/bazel/bazel-0.24.1.x64 /usr/local/bin/bazel

RUN set -e ;\
	apt-get update ;\
	apt-get install -y git build-essential python3-dev python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install wheel setuptools
RUN pip3 install six numpy mock
RUN pip3 install keras_applications==1.0.6 --no-deps
RUN pip3 install keras_preprocessing==1.0.5 --no-deps

RUN set -e ;\
	git clone https://github.com/tensorflow/tensorflow.git ;\
	cd tensorflow ;\
	git checkout ${TF_VER}

# from https://gist.github.com/PatWie/0c915d5be59a518f934392219ca65c3d
# ENV PYTHON_BIN_PATH=$(which ${python_version})
# ENV PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
# ENV PYTHONPATH=${TF_ROOT}/lib
# ENV PYTHON_ARG=${TF_ROOT}/lib

# ENV CUDA_TOOLKIT_PATH=${opt}/cuda/toolkit_8.0/cuda
# ENV CUDNN_INSTALL_PATH=${opt}/cuda/cudnn/6/cuda
  
ENV TF_NEED_GCP=0
ENV TF_NEED_CUDA=0
# ENV TF_CUDA_COMPUTE_CAPABILITIES=6.1,5.2,3.5
ENV TF_NEED_HDFS=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_JEMALLOC=1
ENV TF_ENABLE_XLA=0
ENV TF_NEED_VERBS=0
ENV TF_CUDA_CLANG=0
ENV TF_NEED_MKL=0
ENV TF_DOWNLOAD_MKL=0
ENV TF_NEED_AWS=0
ENV TF_NEED_MPI=0
ENV TF_NEED_GDR=0
ENV TF_NEED_S3=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_SET_ANDROID_WORKSPACE=0
ENV TF_NEED_COMPUTECPP=0
ENV TF_NEED_KAFKA=0
ENV TF_NEED_TENSORRT=0

RUN set -e ;\
	cd tensorflow ;\
	./configure ;\
	bazel build --jobs $(eval "$X_NPROC") --config=opt //tensorflow:libtensorflow.so
