
ROOT=../../..

export VERSION ?= 2.0.0
OSNICK ?= bullseye

#----------------------------------------------------------------------------------------------

S3_URL=redismodules/tensorflow

OS:=$(shell $(ROOT)/opt/readies/bin/platform --os)

ifeq ($(OS),linux)
OS.publish:=$(OS)
ARCH.publish:=$(ARCH)

else ifeq ($(OS),macos)
OS.publish:=darwin
ARCH.publish:=x86_64
endif

STEM=libtensorflowlite-$(OS.publish)

DOCKER_OS.bionic=ubuntu:bionic
DOCKER_OS.bullseye=debian:bullseye-slim
DOCKER_OS=$(DOCKER_OS.$(OSNICK))
 
#----------------------------------------------------------------------------------------------

ifeq ($(OS),linux)

define targets # (1=OP, 2=op)
$(1)_TARGETS :=
$(1)_TARGETS += $(if $(findstring $(X64),1),$(2)_x64)
$(1)_TARGETS += $(if $(findstring $(ARM7),1),$(2)_arm32v7)
$(1)_TARGETS += $(if $(findstring $(ARM8),1),$(2)_arm64v8)

$(1)_TARGETS += $$(if $$(strip $$($(1)_TARGETS)),,$(2)_x64 $(2)_arm32v7 $(2)_arm64v8)
endef

else ifeq ($(OS),macos)

define targets # (1=OP, 2=op)
$(1)_TARGETS := $(2)_x64
endef

endif

$(eval $(call targets,BUILD,build))
$(eval $(call targets,PUBLISH,publish))

#----------------------------------------------------------------------------------------------

define build_x64 # (1=arch, 2=tar-arch)
IID_$(1)=$(1)_$(VERSION).iid
CID_$(1)=$(1)_$(VERSION).cid

build_x64:
	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-$(1):$(VERSION) -f Dockerfile.x64 \
		--build-arg OS=$(DOCKER_OS) $(ROOT)
	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
	@docker cp `cat $$(CID_$(1))`:/build/dest/$(STEM)-$(2)-$(VERSION).tar.gz .

.PHONY: build_x64
endef

define build_arm # (1=arch, 2=tar-arch)
IID_$(1)=$(1)_$(VERSION).iid
CID_$(1)=$(1)_$(VERSION).cid

build_$(1):
	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-$(1):$(VERSION) -f Dockerfile.arm \
		--build-arg ARCH=$(1) $(ROOT)
	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-$(2)-$(VERSION).tar.gz .

.PHONY: build_$(1)
endef

#----------------------------------------------------------------------------------------------

define publish_x64 # (1=arch, 2=tar-arch)
publish_x64:
	@aws s3 cp $(STEM)-$(2)-$(VERSION).tar.gz s3://$(S3_URL)/ --acl public-read

.PHONY: publish_x64
endef

define publish_arm # (1=arch, 2=tar-arch)
publish_$(1):
	@aws s3 cp $(STEM)-$(2)-$(VERSION).tar.gz s3://$(S3_URL)/ --acl public-read

.PHONY: publish_$(1)
endef

#----------------------------------------------------------------------------------------------

all: build publish

ifeq ($(OS),linux)

build: $(BUILD_TARGETS)

$(eval $(call build_x64,x64,x86_64))
$(eval $(call build_arm,arm64v8,arm64))
$(eval $(call build_arm,arm32v7,arm))

ifneq ($(filter publish,$(MAKECMDGOALS)),)
ifeq ($(wildcard $(HOME)/.aws/credentials),)
$(error Please run 'aws configure' and provide it with access credentials)
endif
endif

publish: $(PUBLISH_TARGETS)

$(eval $(call publish_x64,x64,x86_64))
$(eval $(call publish_arm,arm64v8,arm64))
$(eval $(call publish_arm,arm32v7,arm))

help:
	@echo "make [build|publish] [X64=1|ARM7=1|ARM8=1]"

else ifeq ($(OS),macos)

build:
	@VERSION=$(VERSION) ./build.macos
	@mv macos/dest/$(STEM)-$(ARCH.publish)-$(VERSION).tar.gz .

publish: $(PUBLISH_TARGETS)

$(eval $(call publish_x64,x64,x86_64))

help:
	@echo "make [build|publish]"

endif # macos

.PHONY: all build publish help
