
ROOT=../../..

VERSION ?= 1.7.0
OSNICK ?= bionic

REDIS_VER=6.0.9

#----------------------------------------------------------------------------------------------

S3_URL=redismodules/pytorch

OS:=$(shell $(ROOT)/opt/readies/bin/platform --os)
OSNICK:=$(shell $(ROOT)/opt/readies/bin/platform --osnick)
ARCH:=$(shell $(ROOT)/opt/readies/bin/platform --arch)
DEVICE ?= cpu

STEM=libtorch

DOCKER_OS.bionic=ubuntu:bionic
DOCKER_OS.stretch=debian:stretch-slim
DOCKER_OS.buster=debian:buster-slim
DOCKER_OS=$(DOCKER_OS.$(OSNICK))
 
TAR_ARCH.x64=x86_64
TAR_ARCH.arm64v8=arm64
TAR_ARCH.arm32v7=arm

#----------------------------------------------------------------------------------------------

define targets # (1=OP, 2=op)
$(1)_TARGETS :=
$(1)_TARGETS += $(if $(findstring $(X64),1),$(2)_native)
ifeq ($$(CROSS),1)
$(1)_TARGETS += $(if $(findstring $(ARM7),1),$(2)_arm32v7)
$(1)_TARGETS += $(if $(findstring $(ARM8),1),$(2)_arm64v8)
endif

ifeq ($$(CROSS),1)
$(1)_TARGETS += $$(if $$(strip $$($(1)_TARGETS)),,$(2)_native $(2)_arm32v7 $(2)_arm64v8)
else
$(1)_TARGETS += $$(if $$(strip $$($(1)_TARGETS)),,$(2)_native)
endif
endef

$(eval $(call targets,BUILD,build))
$(eval $(call targets,PUBLISH,publish))

#----------------------------------------------------------------------------------------------

DOCKER_BUILD_ARGS += \
	PT_VER=$(VERSION) \
	OSNICK=$(OSNICK) \
	OS=$(DOCKER_OS) \
	ARCH=$(ARCH)

define build_native # (1=arch, 2=tar-arch)
IID_$(1)=$(1)_$(VERSION).iid
CID_$(1)=$(1)_$(VERSION).cid

build_native:
	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM):$(VERSION)-$(DEVICE)-$(1) -f Dockerfile.$(VARIANT) \
		$(foreach A,$(DOCKER_BUILD_ARGS),--build-arg $(A)) .
	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
	@docker cp `cat $$(CID_$(1))`:/build/dest/$(STEM)-$(DEVICE)-$(OS)-$(2)-$(VERSION).tar.gz .

.PHONY: build_native
endef

ifeq ($(CROSS),1)

define build_arm # (1=arch, 2=tar-arch)
IID_$(1)=$(1)_$(VERSION).iid
CID_$(1)=$(1)_$(VERSION).cid

build_$(1):
	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM):$(VERSION)-$(DEVICE)-$(1) -f Dockerfile.$(VARIANT) \
		$(foreach A,$(DOCKER_BUILD_ARGS),--build-arg $(A)) .
	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-$(2)-$(VERSION).tar.gz .

.PHONY: build_$(1)
endef

endif # CROSS

#----------------------------------------------------------------------------------------------

define publish_native # (1=arch, 2=tar-arch)
publish_native:
	@aws s3 cp $(STEM)-$(2)-$(VERSION).tar.gz s3://$(S3_URL)/ --acl public-read

.PHONY: publish_native
endef

ifeq ($(CROSS),1)

define publish_arm # (1=arch, 2=tar-arch)
publish_$(1):
	@aws s3 cp $(STEM)-$(2)-$(VERSION).tar.gz s3://$(S3_URL)/ --acl public-read

.PHONY: publish_$(1)
endef

endif # CROSS

#----------------------------------------------------------------------------------------------

all: build publish

build: $(BUILD_TARGETS)

$(eval $(call build_native,$(ARCH),$(TAR_ARCH.$(ARCH))))
ifeq ($(CROSS),1)
$(eval $(call build_arm,arm64v8,arm64))
$(eval $(call build_arm,arm32v7,arm))
endif

ifneq ($(filter publish,$(MAKECMDGOALS)),)
ifeq ($(wildcard $(HOME)/.aws/credentials),)
$(error Please run 'aws configure' and provide it with access credentials)
endif
endif

publish: $(PUBLISH_TARGETS)

$(eval $(call publish_x64,x64,x86_64))
ifeq ($(CROSS),1)
$(eval $(call publish_arm,arm64v8,arm64))
$(eval $(call publish_arm,arm32v7,arm))
endif

repack:
	@PT_VERSION=$(VERSION) GPU=$(GPU) ./repack.sh

help:
	@echo "make [build|repack|publish] [X64=1|ARM7=1|ARM8=1] [CROSS=1]"

.PHONY: all build repack publish help

