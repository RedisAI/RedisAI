
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

.PHONY: all build clean deps pack test

all: build

version:
	@echo $(PACK_VER)

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
	rm -rf build deps
else
	$(MAKE) -C build clean
endif

deps:
	@echo Fetching dependencies...
	@./get_deps.sh $(DEPS_FLAGS)

pack:
	@[ ! -z `command -v redis-server` ] || { echo "Cannot find redis-server - aborting."; exit 1; }
ifeq ($(wildcard build/pyenv/.),)
	@virtualenv build/pyenv ;\
	. ./build/pyenv/bin/activate ;\
	pip install git+https://github.com/RedisLabs/RAMP
endif
	@echo "Building RAMP file ..."
	@set -e ;\
	. ./build/pyenv/bin/activate ;\
	RAMPOUT=$$(mktemp /tmp/ramp.XXXXXX) ;\
	LD_LIBRARY_PATH=$(PWD)/deps/install/lib \
	ramp pack -m $(PWD)/ramp.yml -o "build/redisai.{os}-{architecture}.{semantic_version}.zip" $(BINDIR)/redisai.so 2> /dev/null | grep '.zip' > $$RAMPOUT ;\
	tail -1 $$RAMPOUT > $(BINDIR)/PACKAGE ;\
	rm -f $RAMPOUT ;\
	echo "Done."
	@echo "Building dependencies file ..."
	@set -e ;\
	PACK_FNAME=$$(basename `cat $(BINDIR)/PACKAGE`) ;\
	ARCHOSVER=$$(echo "$$PACK_FNAME" | sed -e "s/^redisai\.\([^.]*\..*\)\.zip/\1/") ;\
	cd install ;\
	find backends -name "*.so*" | xargs tar pczf redisai-dependencies.$$ARCHOSVER.tgz
	@echo "Done."

test:
ifeq ($(wildcard venv/.),)
	@python3 -m venv test/venv
	@set -e ;\
	cd test ;\
	. venv/bin/activate ;\
	pip -q install git+https://github.com/RedisLabsModules/RLTest@master ;\
	pip -q install -r test_requirements.txt
	@git lfs pull
endif
	@set -e ;\
	cd test ;\
	. venv/bin/activate ;\
	LD_LIBRARY_PATH=$(PWD)/install \
	python3 -m RLTest --test basic_tests.py --module $(BINDIR)/redisai.so
