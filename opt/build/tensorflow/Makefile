REDIS_BAZEL_VERSION?=3.7.2
REDIS_TF_VERSION?=2.5.0

export REDIS_TF_VERSION
export REDIS_BAZEL_VERSION

PRODUCT=tensorflow
DOCKER_ORG=redislabs
VERSION=${REDIS_TF_VERSION}

ROOT=.
UNAME_MACHINE=$(shell uname -m)
ifeq ($(GPU),1)
BACKEND_NAME=lib${PRODUCT}-gpu-${OS}-${UNAME_MACHINE}-${REDIS_TF_VERSION}.tar.gz
else
BACKEND_NAME=lib${PRODUCT}-cpu-${OS}-${UNAME_MACHINE}-${REDIS_TF_VERSION}.tar.gz
endif

READIES=${ROOT}/../../readies

include ../backends.rules
