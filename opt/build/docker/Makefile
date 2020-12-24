
ROOT=../../..
MK.pyver:=3
include $(ROOT)/opt/readies/mk/main

# OSNICK=buster|stretch|bionic|xenial|centos7
OSNICK ?= buster

REPO=redisfab
STEM=$(REPO)/redisai

REDIS_VER:=6.0.9

ART_DIR=$(ROOT)/bin/artifacts
ART_INT_DIR=/var/opt/redislabs/artifacts

FETCH_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0

FETCH_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0

ifeq ($(CPU),1)
VAR=cpu

FETCH_ARGS=$(FETCH_ARGS.$(ARCH))
BUILD_ARGS=$(BUILD_ARGS.$(ARCH))

else ifeq ($(GPU),1)
VAR=gpu
DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.gpu

OS.custom:=1

OS.centos7=centos7
OS.bionic=ubuntu18.04
OS.xenial=ubuntu16.04

FETCH_ARGS=$(FETCH_ARGS.$(ARCH))
BUILD_ARGS=$(BUILD_ARGS.$(ARCH))

else ifeq ($(JETSON),1)
VAR=jetson
DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.jetson

# FETCH_ARGS=WITH_TF=1 WITH_PT=1 WITH_TFLITE=0 WITH_ORT=0
# BUILD_ARGS=WITH_TF=1 WITH_PT=1 WITH_TFLITE=0 WITH_ORT=0

endif

# DOCKER_BUILD_ARGS += \
#	DEPS_ARGS="$(FETCH_ARGS)" \
#	BUILD_ARGS="$(BUILD_ARGS)"

include $(MK)/docker
