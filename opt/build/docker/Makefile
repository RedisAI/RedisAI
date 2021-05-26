# Paths
ROOT:=../../..
READIES:=${ROOT}/opt/readies
PLATFORM=$(shell ${READIES}/bin/platform --arch)
MK:=${READIES}/mk

# set defaults
TEST:=0
PACK:=0
REDISAI_LITE:=0
REDIS_VERSION:=6.0.9

# default dockers to build with docker-compose
DOCKER_SERVICES=\
	xenial\
	bionic

# find the right dockerfile to use, jetson variants end in jetson, gpu end in gpu
DOCKERFILE_NAME=Dockerfile
ifdef VARIANT
DOCKERFILE_NAME=Dockerfile.$(VARIANT)
ifeq($(VARIANT),jetson)
ARCH=jetson
else
ARCH=x64
endif

# defaults to CPU
ifndef VARIANT
VARIANT:=cpu
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

# FUTURE pass in the product and docker orgs as overrides. But this is dependant on the following fix to docker-compose.
# https://github.com/docker/compose/issues/8346
PRODUCT=redisai
DOCKER_ORG=redislabs
DOCKER_BUILD_ARGS=\
	--build-arg REDIS_VER=${REDIS_VERSION} \
	--build-arg VARIANT=${VARIANT} \
	--build-arg REDISAI_LITE=${REDISAI_LITE} \
	--build-arg VERSION=${VERSION} \
	--build-arg TEST=${TEST} \
	--build-arg PACK=${PACK} \
	--build-arg DOCKER_ORG=${DOCKER_ORG}
#	--build-arg PRODUCT=${PRODUCT} \

###### TARGETS ######

build:
	docker-compose build --parallel --pull \
		--build-arg DOCKERFILE_NAME=${DOCKERFILE_NAME} \
		${DOCKER_BUILD_ARGS} \
		${DOCKER_SERVICES}
# gather artifacts when artifacts is set (i.e circleci)
ifeq ($(ARTIFACTS),1)
	$(foreach var, ${DOCKER_SERVICES}, ART_DIR=${ROOT}/bin/artifacts ART_INT_DIR=/var/opt/redislabs/artifacts IMAGE=${DOCKER_ORG}/${PRODUCT}:${VERSION}-${VARIANT}-${ARCH}-${var} ${MK}/docker-collect-artifacts)
endif

# publish our dockers - this assumes you're logged in to dockerhub
publish:
	docker-compose push
