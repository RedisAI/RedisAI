
ROOT=.

VERSION=0.4.0

S3_URL=redismodules/onnxruntime

STEM=onnxruntime-linux

KEEP ?= 1

#----------------------------------------------------------------------------------------------

define targets # (1=OP, 2=op)
$(1)_TARGETS :=
$(1)_TARGETS += $(if $(findstring $(X64),1),$(2)_x64)
$(1)_TARGETS += $(if $(findstring $(ARM7),1),$(2)_arm32v7)
$(1)_TARGETS += $(if $(findstring $(ARM8),1),$(2)_arm64v8)

$(1)_TARGETS += $$(if $$(strip $$($(1)_TARGETS)),,$(2)_x64 $(2)_arm32v7 $(2)_arm64v8)
endef

$(eval $(call targets,BUILD,build))
$(eval $(call targets,PUBLISH,publish))

#----------------------------------------------------------------------------------------------
# arch=x64|arm32v7|arm64v8
# arch_1=x86_64|arm|arm64

define build_arm # (1=arch, 2=arch_1)
IID_$(1)=$(1)_$(VERSION).iid
CID_$(1)=$(1)_$(VERSION).cid

build_$(1):
	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-$(1):$(VERSION) -f Dockerfile.arm \
		--build-arg ARCH=$(1) --build-arg ARCH_FLAG=--$(2) $(ROOT)
	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-$(2)-$(VERSION).tgz .
ifneq ($(KEEP),1)
	# @rm $$(IID_$(1)) $$(CID_$(1))
endif

.PHONY: build_$(1)
endef

#----------------------------------------------------------------------------------------------

define publish_arm # (1=arch, 2=arch_1)
publish_$(1):
	@aws s3 cp $(STEM)-$(2)-$(VERSION).tgz s3://$(S3_URL)/ --acl public-read
ifneq ($(KEEP),1)
	# @rm $(STEM)-$(2)-$(VERSION).tgz
endif

.PHONY: publish_$(1)
endef

#----------------------------------------------------------------------------------------------

all: build publish

build: $(BUILD_TARGETS)

$(eval $(call build_arm,arm64v8,arm64))
$(eval $(call build_arm,arm32v7,arm))

publish: $(PUBLISH_TARGETS)

$(eval $(call publish_arm,arm64v8,arm64))
$(eval $(call publish_arm,arm32v7,arm))

.PHONY: all build publish
