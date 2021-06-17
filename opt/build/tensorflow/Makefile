REDIS_BAZEL_VERSION?=3.7.2
REDIS_TF_VERSION?=2.5.0

export REDIS_TF_VERSION
export REDIS_BAZEL_VERSION

PRODUCT=tensorflow
DOCKER_ORG=redislabs
VERSION=${REDIS_TF_VERSION}

# set here, because otherwise the REDIS_GPU_TYPE bit isn't flipped
ROOT=.
READIES=${ROOT}/../../readies
include ../backends.rules

UNAME_MACHINE=$(shell uname -m)

ifeq ($(GPU),1)
BACKEND_NAME=lib${PRODUCT}-gpu-${OS}-${UNAME_MACHINE}-${REDIS_TF_VERSION}.tar.gz
# future support for others like rocm
REDIS_GPU_TYPE?=cuda
export REDIS_GPU_TYPE
else
BACKEND_NAME=lib${PRODUCT}-cpu-${OS}-${UNAME_MACHINE}-${REDIS_TF_VERSION}.tar.gz
endif

# override
ifeq ($(REDIS_GPU_TYPE),cuda)
build:
	REDIS_ARCH=${ARCH} \
	NOP=${NOP} \
	${READIES}/bin/dockerwrapper \
		-d ${CURDIR}/Dockerfile${DOCKER_SUFFIX} \
		-t ${DEFAULT_DOCKER_TAG} \
        -S ../dockerparts \
		-e REDIS \
		-D "${DOCKER_OPTS}"
		${DOCKER_ARGS}
	docker run --rm --name tensorbuilder --runtime=nvidia --gpus=all -d ${DEFAULT_DOCKER_TAG} sleep 180000
	docker exec tensorbuilder ./configure
	docker exec tensorbuilder bazelisk build --config=${REDIS_GPU_TYPE} --jobs `nproc` //tensorflow:libtensorflow.so
endif
