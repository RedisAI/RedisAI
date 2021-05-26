# Paths
ROOT:=../../..
READIES:=${ROOT}/opt/readies
PLATFORM=$(shell ${READIES}/bin/platform --arch)
MK:=${READIES}/mk

# set defaults
TEST:=0
REDISAI_LITE:=0
REDIS_VERSION:=6.0.9

# by default, we compile only. But if we're gathering artifacts, then we need to pack them first
PACK:=0
ifdef ARTIFACTS
PACK=1
endif

# default dockers to build with docker-compose
DOCKER_SERVICE=\
	xenial

# find the right dockerfile to use, jetson variants end in jetson, gpu end in gpu
DOCKERFILE_NAME=Dockerfile
VARIANT=cpu
ifneq (${VARIANT},cpu)
DOCKERFILE_NAME=Dockerfile.$(VARIANT)
endif

ARCH=${PLATFORM}
ifneq ($(findstring arm,${PLATFORM}),)
ARCH=jetson
endif

# get the release version, unless one is defined
# since in CI, branches with {d}.{d} push, we allow anything here
# but, master is remapped to edge
ifndef VERSION
	VERSION=$(shell git branch --show-current)
	ifeq ($(VERSION),master)
		VERSION=edge
	endif
endif

PRODUCT=redisai
DOCKER_ORG=redislabs
DOCKER_BUILD_ARGS=\
	REDIS_VER=${REDIS_VERSION} \
	VARIANT=${VARIANT} \
	REDISAI_LITE=${REDISAI_LITE} \
	VERSION=${VERSION} \
	TEST=${TEST} \
	PACK=${PACK} \
	DOCKERFILE_NAME=${DOCKERFILE_NAME} \
	DOCKER_ORG=${DOCKER_ORG} \
	PRODUCT=${PRODUCT}

# Given the way nvidia maps their docker images in dockerhub, we need to optionally pass in this variable
# only when building a gpu variant
ifeq ($(VARIANT),gpu)
OS.xenial=ubuntu16.04
OS.bionic=ubuntu18.04
OS=$(OS.$(DOCKER_SERVICE))
DOCKER_BUILD_ARGS += OS=${OS}
endif

###### TARGETS ######
build:
	$(shell echo ${DOCKER_BUILD_ARGS} | tr " " "\n" > .generated.env)
	@cat .generated.env
	docker-compose --env-file .generated.env build --pull ${DOCKER_SERVICE}
# gather artifacts when artifacts is set (i.e circleci)
ifeq ($(ARTIFACTS),1)
	$(foreach var, ${DOCKER_SERVICE}, ART_DIR=${ROOT}/bin/artifacts ART_INT_DIR=/var/opt/redislabs/artifacts IMAGE=${DOCKER_ORG}/${PRODUCT}:${VERSION}-${VARIANT}-${ARCH}-${var} ${MK}/docker-collect-artifacts)
endif

# publish our dockers - this assumes you're logged in to dockerhub
publish:
	docker-compose push
