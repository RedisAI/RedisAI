override GPU:=$(or $(findstring $(CUDA),1),$(findstring $(GPU),1))

ifeq ($(GPU),1)
ifeq ($(CPU),1)
$(error CPU=1 and GPU=1 (or CUDA=1) are conflicting)
endif
DEPS_FLAGS=gpu
ENGINE=gpu
else
DEPS_FLAGS=cpu
ENGINE=cpu
endif

export REDIS_ENT_LIB_PATH=/opt/redislabs/lib

GIT_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
GIT_COMMIT:=$(shell git describe --always --abbrev=7 --dirty="+")

ifeq ($(VERSION),)
PACK_VER:=$(GIT_BRANCH)-$(GIT_COMMIT)
else
PACK_VER:=$(VERSION)
endif

BINDIR=$(PWD)/install-$(ENGINE)

BACKENDS_PATH ?= $(BINDIR)/backends

#----------------------------------------------------------------------------------------------

setup:
	@echo Setting up system...
	$(SHOW)./deps/readies/bin/getpy
	$(SHOW)./system-setup.py

.PHONY: all setup build clean deps pack pack_ramp pack_deps test

all: build

ifeq ($(DEBUG),1)
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Debug
endif

build:
ifeq ($(wildcard build/Makefile),)
	mkdir -p build
	cd build; \
	cmake -DENGINE=$(ENGINE) -DDEPS_PATH=../deps/install-$(ENGINE) $(CMAKE_FLAGS) ..
endif
	$(MAKE) -C build
	$(MAKE) -C build install

clean:
ifeq ($(ALL),1)
	rm -rf build install deps/dlpack deps/install-$(ENGINE) deps/*.tar.gz deps/*.zip deps/*.tgz
else
	$(MAKE) -C build clean
endif

deps fetch:
	@echo Fetching dependencies...
	@./get_deps.sh $(DEPS_FLAGS)

pack: pack_ramp pack_deps

pack_ramp:
	@[ ! -z `command -v redis-server` ] || { echo "Cannot find redis-server - aborting."; exit 1; }
	@echo "Building RAMP file ..."
	@set -e ;\
	RAMPOUT=$$(mktemp /tmp/ramp.XXXXXX) ;\
	LD_LIBRARY_PATH=$(PWD)/deps/install/lib \
	ramp pack -m $(PWD)/ramp.yml -o "build/redisai.{os}-{architecture}.{semantic_version}.zip" $(BINDIR)/redisai.so 2> /dev/null | grep '.zip' > $$RAMPOUT ;\
	tail -1 $$RAMPOUT > $(BINDIR)/PACKAGE ;\
	rm -f $RAMPOUT ;\
	echo "Done."

pack_deps: pack_ramp
	@echo "Building dependencies file ..."
	@set -e ;\
	PACK_FNAME=$$(basename `cat $(BINDIR)/PACKAGE`) ;\
	ARCHOSVER=$$(echo "$$PACK_FNAME" | sed -e "s/^redisai\.\([^.]*\..*\)\.zip/\1/") ;\
	cd install ;\
	find backends -name "*.so*" | xargs tar pczf redisai-dependencies.$$ARCHOSVER-$(ENGINE).tgz ;\
	echo "Done."

test:
	@git lfs install
	@git lfs pull
	@set -e ;\
	cd test ;\
	python3 -m RLTest $(TEST_ARGS) --test basic_tests.py --module $(BINDIR)/redisai.so
