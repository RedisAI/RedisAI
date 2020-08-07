
ROOT=../../..
MK.pyver:=3
include $(ROOT)/opt/readies/mk/main

# OSNICK=buster|stretch|bionic|xenial|centos7
OSNICK ?= buster

REPO=redisfab
STEM=$(REPO)/redisai

ART_DIR=$(ROOT)/bin/artifacts
ART_INT_DIR=/var/opt/redislabs/artifacts

FETCH_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0

FETCH_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0

ifeq ($(CPU),1)
VAR=cpu

else ifeq ($(GPU),1)
VAR=gpu
DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.gpu

OS.custom:=1

OS.centos7=centos7
OS.bionic=ubuntu18.04
OS.xenial=ubuntu16.04

else
VAR=cpu
endif

DOCKER_BUILD_ARGS += \
	DEPS_ARGS="$(FETCH_ARGS.$(ARCH))" \
	BUILD_ARGS="$(BUILD_ARGS.$(ARCH))"

include $(MK)/docker
