REDIS_ONNX_VERSION?=1.9.0
REDIS_ONNX_REPO?=https://github.com/microsoft/onnxruntime

PRODUCT=onnxruntime
VERSION=${REDIS_ONNX_VERSION}
REDIS_CUDA_VERSION=11.0-cudnn8

# anything prefixed with REDIS, and exported is available in the docker build
export REDIS_ONNX_VERSION
export REDIS_ONNX_REPO
export REDIS_CUDA_VERSION

OSNICK=xenial

ROOT=.
READIES=${ROOT}/../../readies

include ../backends.rules
