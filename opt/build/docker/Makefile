# Feel free to change these
PRODUCT=redisai
DOCKER_ORG=redislabs
REDIS_VERSION=6.2.7
REDIS_CUDA_VERSION=11.3.1-cudnn8
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

# set gpu arguments
ifeq ($(GPU),1)
DOCKER_ARGS = gpu_build=1
DOCKER_SUFFIX=gpu
DEFAULT_TAG=${DOCKER_ORG}/${PRODUCT}:${VERSION}-gpu-${OSNICK}
endif

# remap ubuntu versions because of nvidia cuda
ifeq ($(OSNICK),xenial)
REDIS_CUDA_MAPVERSION=ubuntu16.04
endif
ifeq ($(OSNICK),bionic)
REDIS_CUDA_MAPVERSION=ubuntu18.04
endif
ifeq ($(OSNICK),centos8)
REDIS_CUDA_MAPVERSION=centos8
endif

DOCKERWRAPPER_EXTRA_VARS=\
	REDISAI_LITE=${REDISAI_LITE} \
	REDIS_CUDA_VERSION=${REDIS_CUDA_VERSION} \
	REDIS_CUDA_MAPVERSION=${REDIS_CUDA_MAPVERSION}

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
	GPU=1						# if set, build the GPU docker
endef
#------------------------------------------------------------------------------

# rules, for building
include ${READIES}/mk/docker.rules
