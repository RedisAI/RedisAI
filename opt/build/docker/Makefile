# Valid options during make include:
# VERSION: Only if you want to override the default
# REDIS_VER: version of redis
# DOCKER_ARCH: x64|arm|jetson
# DOCKER_ORG: docker organization for publishing
# DOCKER_SERVICES: the specific dockers to build
# PUBLISH: Defaults to 1, but if PUBLISH=0 then do not publish

ROOT=../../..

VERSION=$(shell ../../../opt/readies/bin/generateversion a|cut -d '-' -f 1-1)
REDIS_VER:=6.0.12

# default, do not publish
PUBLISH=1

DOCKER_ARCH=$(shell ../../../opt/readies/bin/platform --arch)
ifeq ($(VERSION),master)
VERSION=edge
else
ifeq ($(VERSION),)
VERSION=edge
endif
endif

# can be xenial buster bionic focal centos7...
DOCKER_SERVICES=\
	xenial\
	bionic

# cpu|gpu
VARIANT=cpu
ifeq ($(VARIANT),gpu)
DOCKERFILE_NAME=Dockerfile.gpu
else ifeq ($(VARIANT),jetson)
DOCKERFILE_NAME=Dockerfile.jetson
else
ifeq ($(DOCKER_ARCH),arm64v8)
DOCKERFILE_NAME=Dockerfile.arm
else
DOCKERFILE_NAME=Dockerfile
endif
endif
DOCKERFILE_NAME=Dockerfile

CLEANFLAGS=
ifdef NOCLEAN
CLEANFLAGS=--no-rm --no-cache
endif

# run tests in the docker
ifndef TEST
TEST=0
endif

# pack artifacts
ifndef PACK
PACK=0
endif

# override, to not push to the redislabs docker
DOCKER_ORG=redislabs

.NOTPARALLEL:

# build whatever docker is specified in variant (defaults to CPU)
docker::
	$(call build_docker,$(VARIANT),${DOCKERFILE_NAME})

# build cpu dockers
docker_cpu:
	$(call build_docker,cpu,${DOCKERFILE_NAME})

# build gpu dockers
docker_gpu:
	$(call build_docker,gpu,Dockerfile.gpu)

dockers:: docker_cpu docker_gpu

# call with cpu,Dockerfile (for example)
define build_docker
	@rm -f .env
	@@echo "VERSION=${VERSION}" >> .env
	@@echo "REDIS_VERSION=${REDIS_VER}" >> .env
	@@echo "VARIANT=${1}" >> .env
	@@echo "DOCKER_ARCH=${DOCKER_ARCH}" >> .env
	@@echo "DOCKERFILE_NAME=${2}" >> .env
	@@echo "DOCKER_ORG=${DOCKER_ORG}" >> .env
	@@echo "PACK=${PACK}" >> .env
	@@echo "TEST=${TEST}" >> .env
	@@cat .env
	docker-compose build --parallel ${DOCKER_SERVICES} ${CLEANFLAGS}
	docker rmi `docker images -f "dangling=true"`
$(if $(filter ${PUBLISH},1),@echo -n "",echo docker-compose push ${DOCKER_SERVICES})
endef

# if you've been testing and building locally, and just want to push
publish_dockers::
	docker-compose push ${DOCKER_SERVICES}
.PHONY: publish_dockers
