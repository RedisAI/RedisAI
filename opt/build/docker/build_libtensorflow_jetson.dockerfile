ARG BUILD_IMAGE=nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf2.2-py3
FROM ${BUILD_IMAGE}

ARG CUDA_VERSION="10.2"
ARG CUDA_TOOLKIT_PATH="/usr/local/cuda"
ARG CUDNN_VERSION="8"
ARG PY_VERSION_SUFFIX=""
ARG TF_BRANCH="r2.3"
ARG TF_TENSORRT_VERSION="7.2"
ARG TF_NCCL_VERSION=""

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

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-8-jdk \
    python${PY_VERSION_SUFFIX} \
    python${PY_VERSION_SUFFIX}-dev \
    python${PY_VERSION_SUFFIX}-pip \
    swig

RUN cd / && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd /tensorflow && \
    git checkout ${TF_BRANCH}

WORKDIR /tensorflow

# Set environment variables for configure.
ENV PYTHON_BIN_PATH=python${PY_VERSION_SUFFIX} \
    TF_NEED_CUDA=1 \
    TF_NEED_TENSORRT=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    TF_TENSORRT_VERSION=${TF_TENSORRT_VERSION} \
    TF_CUDA_VERSION=${CUDA_VERSION} \
    TF_NCCL_VERSION=${TF_NCCL_VERSION} \
    TF_CUDNN_VERSION=${CUDNN_VERSION} \
    TF_CUDA_COMPUTE_CAPABILITIES=5.3 \
    CUDA_TOOLKIT_PATH=""

RUN yes "" | ./configure && \
    bazel build --config=elinux_aarch64 --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        --config=v2 --config=noaws --config=nogcp --config=cuda --config=nonccl --config=nohdfs \
        --config opt //tensorflow/tools/lib_package:libtensorflow 

WORKDIR /root
