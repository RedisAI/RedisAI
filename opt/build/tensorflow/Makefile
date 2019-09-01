
ROOT=../..

X64_VERSION=1.13.1
ARM_VERSION=1.14.0

S3_URL=redismodules/tensorflow

STEM=libtensorflow-cpu-linux

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

define build_arm # (1=arch, 2=tar-arch,3=version)
IID_$(1)=$(1)_$(3).iid
CID_$(1)=$(1)_$(3).cid

build_$(1):
	@docker build --iidfile $$(IID_$(1)) -t redisfab/$(STEM)-$(1):$(3) --build-arg ARCH=$(1) $(ROOT)
	@docker create --cidfile $$(CID_$(1)) `cat $$(IID_$(1))`
	@docker cp `cat $$(CID_$(1))`:/build/$(STEM)-$(2)-$(3).tar.gz .

.PHONY: build_$(1)
endef

#----------------------------------------------------------------------------------------------

define publish_arm # (1=arch, 2=tar-arch,3=version)
publish_$(1):
	@aws s3 cp $(STEM)-$(2)-$(3).tar.gz s3://$(S3_URL)/ --acl public-read

.PHONY: publish_$(1)
endef

#----------------------------------------------------------------------------------------------

all: build publish

build: $(BUILD_TARGETS)

$(eval $(call build_arm,arm64v8,arm64,$(ARM_VERSION)))
$(eval $(call build_arm,arm32v7,arm,$(ARM_VERSION)))

publish: $(PUBLISH_TARGETS)

$(eval $(call publish_arm,arm64v8,arm64,$(ARM_VERSION)))
$(eval $(call publish_arm,arm32v7,arm,$(ARM_VERSION)))

.PHONY: all build publish
