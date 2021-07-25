REDIS_ONNX_VERSION?=rel-1.7.2_RCE
REDIS_ONNX_REPO?=https://github.com/redisai/onnxruntime

export REDIS_ONNX_VERSION
export REDIS_ONNX_REPO

PRODUCT=onnxruntime
DOCKER_ORG=redislabs
REDIS_BACKEND_VERSION?=${REDIS_ONNX_VERSION}
export REDIS_VERSION

ROOT=.

READIES=${ROOT}/../../readies

include ../backends.rules
