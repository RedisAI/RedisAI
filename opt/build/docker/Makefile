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
DOCKER_SERVICES=\
	xenial\
	bionic

# find the right dockerfile to use, jetson variants end in jetson, gpu end in gpu
DOCKERFILE_NAME=Dockerfile
VARIANT=cpu
ifneq (${VARIANT},cpu)
DOCKERFILE_NAME=Dockerfile.$(VARIANT)
endif

ARCH=${PLATFORM}
#r=($(findstring arm,${PLATFORM})
#ifeq ($r,)
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

# FUTURE pass in the product and docker orgs as overrides. But this is dependant on the following fix to docker-compose.
# https://github.com/docker/compose/issues/8346
PRODUCT=redisai
DOCKER_ORG=redislabs
DOCKER_BUILD_ARGS=\
	REDIS_VER=${REDIS_VERSION} \
	VARIANT=${VARIANT} \
	REDISAI_LITE=${REDISAI_LITE} \
	VERSION=${VERSION} \
	TEST=${TEST} \
	PACK=${PACK} \
	DOCKER_ORG=${DOCKER_ORG} \
	PRODUCT=${PRODUCT}

###### TARGETS ######
build:
	$(shell echo ${DOCKER_BUILD_ARGS} | tr " " "\n" > .generated.env)
	@cat .generated.env
	docker-compose --env-file .generated.env build --parallel --pull ${DOCKER_SERVICES}
# gather artifacts when artifacts is set (i.e circleci)
ifeq ($(ARTIFACTS),1)
	$(foreach var, ${DOCKER_SERVICES}, ART_DIR=${ROOT}/bin/artifacts ART_INT_DIR=/var/opt/redislabs/artifacts IMAGE=${DOCKER_ORG}/${PRODUCT}:${VERSION}-${VARIANT}-${ARCH}-${var} ${MK}/docker-collect-artifacts)
endif

# publish our dockers - this assumes you're logged in to dockerhub
publish:
	docker-compose push
