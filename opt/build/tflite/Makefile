ROOT=../../..

export VERSION ?= 2.4.0

#----------------------------------------------------------------------------------------------

S3_URL=redismodules/tensorflow

OS:=$(shell $(ROOT)/opt/readies/bin/platform --os)
ARCH:=$(shell $(ROOT)/opt/readies/bin/platform --arch)

# non-arm linux
ifeq ($(OS),linux)
ifeq ($(ARCH),x64)
OS.publish:=$(OS)
ARCH.publish:=x86_64

BUILD_TARGETS:=build_x64
PUBLISH_TARGETS:=publish_x64
else ifeq ($(ARCH),arm64v8)
BUILD_TARGETS:=build_arm
PUBLISH_TARGETS:=publish_arm
else ifeq ($(ARCH),)
BUILD_TARGETS:=
PUBLISH_TARGETS:=
endif

# mac
else ifeq ($(OS),macos)
OS.publish:=darwin
ARCH.publish:=x86_64
endif

STEM=libtensorflowlite-$(OS.publish)

# x86 linux ------------------------------------------------------------------------------------------

build_x64:
	-@rm *.iid *.cid
	docker build --iidfile x64.iid -t redisfab/$(STEM)-x86_64:$(VERSION) -f Dockerfile.x64 --build-arg TFLITE_VERSION=$(VERSION) $(ROOT)
	docker create --cidfile x64.cid `cat x64.iid`
	docker cp `cat x64.cid`:/tflite/tensorflow-$(VERSION)/tmp/$(STEM)-$(ARCH.publish)-$(VERSION).tar.gz .
.PHONY: build_x64

publish_x64:
	@aws s3 cp $(STEM)-x86_64-$(VERSION).tar.gz s3://$(S3_URL)/ --acl public-read
.PHONY: publish_x64

# arm linux ------------------------------------------------------------------------------------------

ifeq ($(ARCH),arm64v8)
ARCH.publish:=arm64
DOCKERBASE=redisfab/ubuntu1804-$(ARCH.publish)-bazel3.1.0-jetson:latest
endif

build_arm:
	@cd ../../../ ; docker run -v `pwd`/opt/build/tflite:/tflite -v /etc/alternatives:/etc/alternatives -v /usr/lib:/usr/lib -v /usr/local:/usr/local -v /usr/include:/usr/include ${DOCKERBASE} /tflite/build.sh $(ARCH.publish) $(VERSION)

.PHONY: build_arm

publish_arm:
	@aws s3 cp `pwd`/tensorflow-$(VERSION)/tmp/$(STEM)$(OS)-$(ARCH.publish)-$(VERSION).tar.gz s3://$(S3_URL)/ --acl public-read

.PHONY: publish_arm

#----------------------------------------------------------------------------------------------

all: build publish

ifneq ($(filter publish,$(MAKECMDGOALS)),)
ifeq ($(wildcard $(HOME)/.aws/credentials),)
$(error Please run 'aws configure' and provide it with access credentials)
endif
endif

help:
	@echo "make [build|publish]"

build: $(BUILD_TARGETS)
publish: $(PUBLISH_TARGETS)

.PHONY: all build publish help
