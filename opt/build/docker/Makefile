# Feel free to change these
PRODUCT=redisai
DOCKER_ORG=redislabs
export REDIS_VERSION=6.2.4
export REDIS_CUDA_VERSION=11.0-cudnn8
export REDISAI_LITE=0

PACK=1 # to fetch the artifacts
TEST=1 # because dockers
OSNICK ?= bionic

DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-cpu-${OSNICK}

# jetson flavour
ifdef REDIS_JETSON
export REDIS_JETSON=1
export REDIS_CUDA_VERSION=10.2-cudnn7
DOCKER_SUFFIX=.jetson
DOCKER_ARGS=gpu_build=1
DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-gpu-jetson
endif

# set gpu arguments
ifeq ($(GPU),1)
DOCKER_ARGS = gpu_build=1
DOCKER_SUFFIX=.gpu
DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-gpu-${OSNICK}
endif

# remap ubuntu versions because of nvidia cuda
ifeq ($(OSNICK),xenial)
export REDIS_CUDA_UBUNTUVERSION=16.04
endif
ifeq ($(OSNICK),bionic)
export REDIS_CUDA_UBUNTUVERSION=18.04
endif

### Defaults ###
ROOT=../../..
READIES=${ROOT}/opt/readies

# rules, for building
include ${READIES}/mk/docker.rules
