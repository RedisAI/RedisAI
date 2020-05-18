
ROOT=../../..
include $(ROOT)/opt/readies/mk/main

# OSNICK=buster|stretch|bionic|xenial|centos7
OSNICK ?= buster

REPO=redisfab
STEM=$(REPO)/redisai-cpu

FETCH_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0

FETCH_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0

DOCKER_BUILD_ARGS += \
	DEPS_ARGS="$(FETCH_ARGS.$(ARCH))" \
	BUILD_ARGS="$(BUILD_ARGS.$(ARCH))"

include $(MK)/docker
