ifeq ($(CUDA),1)
DEPS_FLAGS=
else
DEPS_FLAGS=cpu
endif

export REDIS_ENT_LIB_PATH=/opt/redislabs/lib

GIT_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
GIT_COMMIT:=$(shell git describe --always --abbrev=7 --dirty="+")

ifeq ($(VERSION),)
PACK_VER:=$(GIT_BRANCH)-$(GIT_COMMIT)
else
PACK_VER:=$(VERSION)
endif

BINDIR=$(PWD)/install

BACKENDS_PATH ?= $(BINDIR)/backends

.PHONY: all build clean deps pack pack_ramp pack_deps test

all: build

build:
ifeq ($(wildcard build/.),)
	mkdir -p build
	cd build; \
	cmake -DDEPS_PATH=../deps/install ..
endif
	$(MAKE) -C build
	$(MAKE) -C build install

clean:
ifeq ($(ALL),1)
	rm -rf build install deps/dlpack deps/install deps/*.tar.gz deps/*.zip deps/*.tgz
	
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
	find backends -name "*.so*" | xargs tar pczf redisai-dependencies.$$ARCHOSVER.tgz ;\
	echo "Done."

test:
	@git lfs install
	@git lfs pull
	@set -e ;\
	cd test ;\
	python3 -m RLTest $(TEST_ARGS) --test basic_tests.py --module $(BINDIR)/redisai.so
