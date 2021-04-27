
ROOT=../../..
MK.pyver:=3
include $(ROOT)/opt/readies/mk/main

# OSNICK=buster|stretch|bionic|xenial|centos7
OSNICK ?= bionic

REPO=redisai

# LATEST_BRANCH=1.2
INT_BRANCHES=1.2

REDIS_VER:=6.0.9

export ART_DIR=$(ROOT)/bin/artifacts
export ART_INT_DIR=/var/opt/redislabs/artifacts

#----------------------------------------------------------------------------------------------

ifeq ($(CPU),1)
VARIANT=cpu

else ifeq ($(GPU),1)
VARIANT=gpu
DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.gpu

OS.custom:=1

OS.centos7=centos7
OS.bionic=ubuntu18.04
OS.xenial=ubuntu16.04

else ifeq ($(JETSON),1)
VARIANT=jetson
DOCKERFILE_STEM ?= $(ROOT)/Dockerfile.jetson

endif

#----------------------------------------------------------------------------------------------

include $(MK)/docker.native
