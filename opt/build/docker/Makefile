
ROOT=../../..
MK.pyver:=3

#include $(ROOT)/opt/readies/mk/main
#
## OSNICK=buster|stretch|bionic|xenial|centos7
#OSNICK ?= buster
#
#REPO=redisai
#
## LATEST_BRANCH=1.2
#INT_BRANCHES=1.2
#
REDIS_VER:=6.0.9
VERSION=$(shell ../../../opt/readies/bin/generateversion a|cut -d '-' -f 1-1)
DOCKER_ARCH:=x64
ifeq ($(VERSION),master)
VERSION=edge
endif

DOCKER_SERVICES=\
	xenial\
	buster\
	bionic

# cpu|gpu
VARIANT:=cpu
ifeq ($(VARIANT),gpu)
DOCKERFILE_NAME=Dockerfile.gpu
else ifeq ($(VARIANT),jetson)
DOCKERFILE_NAME=Dockerfile.jetson
else
DOCKERFILE_NAME=Dockerfile
endif

docker_cpu:
	$(call build_docker,cpu,${DOCKERFILE_NAME})

docker_gpu:
	$(call build_docker,gpu,Dockerfile.gpu)

docker::
	$(call build_docker,${VARIANT},${DOCKERFILE_NAME})

dockers:: docker_cpu docker_gpu

# call with cpu,Dockerfile (for example)
define build_docker
	@rm -f .env
	@echo "VERSION=${VERSION}" >> .env
	@echo "REDIS_VERSION=${REDIS_VER}" >> .env
	@echo "VARIANT=${1}" >> .env
	@echo "DOCKER_ARCH=${DOCKER_ARCH}" >> .env
	@echo "DOCKERFILE_NAME=${2}" >> .env
	cat .env
	docker-compose build --parallel ${DOCKER_SERVICES}
endef

docker_publish: dockers
	docker-compose push ${DOCKER_SERVICES}
#
#export ART_DIR=$(ROOT)/bin/artifacts
#export ART_INT_DIR=/var/opt/redislabs/artifacts
#
##----------------------------------------------------------------------------------------------
#
#ifeq ($(CPU),1)
#VARIANT=cpu
#
#else ifeq ($(GPU),1)
#VARIANT=gpu
#DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.gpu
#
#OS.custom:=1
#
#OS.centos7=centos7
#OS.bionic=ubuntu18.04
#OS.xenial=ubuntu16.04
#
#else ifeq ($(JETSON),1)
#VARIANT=jetson
#DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.jetson
#
#endif
#
##----------------------------------------------------------------------------------------------
#
#include $(MK)/docker.native
