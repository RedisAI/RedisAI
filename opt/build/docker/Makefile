# Feel free to change these
PRODUCT=redisai
DOCKER_ORG=redislabs
REDIS_VERSION=6.2.4
REDIS_CUDA_VERSION=11.0-cudnn8
REDISAI_LITE=0

PACK=1 # to fetch the artifacts

DOCKER_OPTS=  # set, to pass custom options to docker, but remember to quote them
OSNICK ?= bionic

# set to pass multiple directories as docker sources to include in templates
DOCKER_SOURCES=""

# set, to generate a file named Dockerfile<suffix>, i.e useful when you want to run things in parallel
DOCKER_SUFFIX=

# Add space delimited docker tags, and they'll all be applied
DOCKER_TAGS=

DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-cpu-${OSNICK}

ifeq ($(GPU),1)
DOCKER_ARGS = gpu_build=1
DOCKER_SUFFIX=gpu
OSNICK = bionic
DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-gpu-${OSNICK}
endif

DOCKERWRAPPER_EXTRA_VARS=\
	REDISAI_LITE=${REDISAI_LITE} \
	REDIS_CUDA_VERSION=${REDIS_CUDA_VERSION}

### Defaults ###
ROOT=../../..
READIES=${ROOT}/opt/readies

# rules, for building
include ${READIES}/mk/docker.rules
