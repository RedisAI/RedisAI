# Feel free to change these
PRODUCT=redisai
DOCKER_ORG=redislabs
export REDIS_VERSION=6.2.4
export REDIS_CUDA_VERSION=11.0-cudnn8
export REDISAI_LITE=0

PACK=1 # to fetch the artifacts
OSNICK ?= bionic

DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-cpu-${OSNICK}

# jetson flavour
ifdef REDIS_JETSON
export REDIS_JETSON=1
export REDIS_CUDA_VERSION=10.2-cudnn7
DOCKER_SUFFIX=.jetson
DOCKER_ARGS=-g gpu_build=1
DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-jetson
endif

# set gpu arguments
ifeq ($(GPU),1)
DOCKER_ARGS = gpu_build=1
DOCKER_SUFFIX=gpu
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

#------------------------------------------------------------------------------
define HELP
make build		# build and optionally publish the docker, from a template file
	OSNICK=<>					# base operating system
	REDISAI_LITE=1					# enable RedisAI lite builds
	DOCKER_SUFFIX=<>				# optional suffix for the generated dockerfile
	DOCKER_ARGS=FOO=BAR				# key-value pairs of variables to pass into the docker build
	PACK=1						# fetch generated artifacts
	DOCKER_TAGS=a,b,c				# tags to append and push to dockerhub
	DEFAULT_TAG=redislabs/redisai			# default docker tag to build, and push
	TEST=1						# run tests, if specified
	VERSION=x.y.z					# set the docker version
	NOP=1						# set to echo files in docker generation, and not run
	DOCKER_SOURCES=/a/path				# append paths to the template generator
	PUBLISH=1					# if set, push to dockerhub (requires docker login)
	DOCKER_OPTS=XXX					# Options to pass to the docker build command
endef
#------------------------------------------------------------------------------

# rules, for building
include ${READIES}/mk/docker.rules
