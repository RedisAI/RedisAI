ARG BUILD_IMAGE=nvcr.io/nvidia/tensorflow:20.09-tf2-py3
FROM ${BUILD_IMAGE}

ARG CUDA_VERSION="10.2"
ARG CUDNN_VERSION="8"
ARG PY_VERSION_SUFFIX=""
ARG TF_BRANCH=r2.3

RUN wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-linux-arm64
RUN chmod +x bazel-3.6.0-linux-arm64
RUN mv bazel-3.6.0-linux-arm64 /usr/bin/bazel

# Running bazel inside a `docker build` command causes trouble, cf:
# https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc

RUN sudo apt install -y python3-dev python3-pip
RUN pip3 install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'
RUN pip3 install -U --user keras_applications --no-deps
RUN pip3 install -U --user keras_preprocessing --no-deps

RUN cd / && \
    git clone http://github.com/tensorflow/tensorflow && \
    cd /tensorflow && \
    git checkout ${TF_BRANCH} \
WORKDIR /tensorflow

# Set environment variables for configure.
ENV PYTHON_BIN_PATH=python${PY_VERSION_SUFFIX} \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=${CUDA_VERSION} \
    TF_CUDNN_VERSION=${CUDNN_VERSION} \
    TF_CUDA_COMPUTE_CAPABILITIES=5.3

RUN yes "" | ./configure && \
    bazel build --config opt //tensorflow/tools/lib_package:libtensorflow --config=v2 --config=noaws --config=nogcp --config=cuda --config=nonccl --config=nohdfs

WORKDIR /root
