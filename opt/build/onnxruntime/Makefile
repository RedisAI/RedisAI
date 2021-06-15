export REDIS_ONNX_VERSION = 1.8.0
export REDIS_ONNX_REPO = https://github.com/microsoft/onnxruntime

PRODUCT=onnxruntime
DOCKER_ORG=redislabs
VERSION=${REDIS_ONNX_VERSION}

ROOT=.

READIES=${ROOT}/../../readies

include ../backends.rules

#VERSION ?= 1.8
#OSNICK ?= buster
#
##----------------------------------------------------------------------------------------------
#
#S3_URL=redismodules/onnxruntime
#
#STEM=onnxruntime-linux
#
#KEEP ?= 1
#
#DOCKER_OS.bionic=ubuntu:bionic
#DOCKER_OS.stretch=debian:stretch-slim
#DOCKER_OS.buster=debian:buster-slim
#DOCKER_OS=$(DOCKER_OS.$(OSNICK))
#
##----------------------------------------------------------------------------------------------
#
#define targets # (1=OP, 2=op)
#$(1)_TARGETS :=
#$(1)_TARGETS += $(if $(findstring $(X64),1),$(2)_x64)
#$(1)_TARGETS += $(if $(findstring $(ARM7),1),$(2)_arm32v7)
#$(1)_TARGETS += $(if $(findstring $(ARM8),1),$(2)_arm64v8)
#
#$(1)_TARGETS += $$(if $$(strip $$($(1)_TARGETS)),,$(2)_x64 $(2)_arm32v7 $(2)_arm64v8)
#endef
#
#$(eval $(call targets,BUILD,build))
#$(eval $(call targets,PUBLISH,publish))
#
##----------------------------------------------------------------------------------------------
## arch=x64|arm32v7|arm64v8
## tar_arch=x86_64|arm|arm64
#
#define build_x64 # (1=arch, 2=tar-arch)
#IID_$(1)=$(1)_$(VERSION).iid
#CID_$(1)=$(1)_$(VERSION).cid
#
#build_x64:
#	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-x64:$(VERSION) -f Dockerfile.x64 \
#		--build-arg OS=$(DOCKER_OS) $(ROOT)
#	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
#	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-x64-$(VERSION).tgz .
#
#.PHONY: build_x64
#endef
#
#define build_x64-gpu # (1=arch, 2=tar-arch)
#IID_$(1)=$(1)_$(VERSION).iid
#CID_$(1)=$(1)_$(VERSION).cid
#
#build_x64-gpu:
#	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-$(1):$(VERSION) -f Dockerfile.x64-gpu \
#		--build-arg OS=$(DOCKER_OS) $(ROOT)
#	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
#	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-$(1)-$(VERSION).tgz .
#
#.PHONY: build_x64
#endef
#
#define build_arm # (1=arch, 2=tar-arch)
#IID_$(1)=$(1)_$(VERSION).iid
#CID_$(1)=$(1)_$(VERSION).cid
#
#build_$(1):
#	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-$(1):$(VERSION) -f Dockerfile.arm \
#		--build-arg ARCH=$(1) --build-arg ARCH_FLAG=--$(2) $(ROOT)
#	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
#	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-$(2)-$(VERSION).tgz .
#ifneq ($(KEEP),1)
#	# @rm $$(IID_$(1)) $$(CID_$(1))
#endif
#
#.PHONY: build_$(1)
#endef
#
##----------------------------------------------------------------------------------------------
#
#define publish_x64 # (1=arch, 2=tar-arch)
#publish_x64:
#	@aws s3 cp $(STEM)-$(2)-$(VERSION).tgz s3://$(S3_URL)/ --acl public-read
#
#.PHONY: publish_x64
#endef
#
#define publish_x64-gpu # (1=arch, 2=tar-arch)
#publish_x64-gpu:
#	@aws s3 cp $(STEM)-$(2)-$(VERSION).tgz s3://$(S3_URL)/ --acl public-read
#
#.PHONY: publish_x64-gpu
#endef
#
#define publish_arm # (1=arch, 2=tar-arch)
#publish_$(1):
#	@aws s3 cp $(STEM)-$(2)-$(VERSION).tgz s3://$(S3_URL)/ --acl public-read
#ifneq ($(KEEP),1)
#	# @rm $(STEM)-$(2)-$(VERSION).tgz
#endif
#
#.PHONY: publish_$(1)
#endef
#
##----------------------------------------------------------------------------------------------
#
#all: build publish
#
#build: $(BUILD_TARGETS)
#
#$(eval $(call build_x64,x64,x86_64))
#$(eval $(call build_x64-gpu,x64-gpu,x86_64-gpu))
#$(eval $(call build_arm,arm64v8,arm64))
#$(eval $(call build_arm,arm32v7,arm))
#
#ifneq ($(filter publish,$(MAKECMDGOALS)),)
#ifeq ($(wildcard $(HOME)/.aws/credentials),)
#$(error Please run 'aws configure' and provide it with access credentials)
#endif
#endif
#
#publish: $(PUBLISH_TARGETS)
#
#$(eval $(call publish_x64,x64,x64))
#$(eval $(call publish_x64-gpu,x64-gpu,x64-gpu))
#$(eval $(call publish_arm,arm64v8,arm64))
#$(eval $(call publish_arm,arm32v7,arm))
#
#
#help:
#	@echo "make [build|publish] [X64=1|ARM7=1|ARM8=1]"
#
#.PHONY: all build publish help
