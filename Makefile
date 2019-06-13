
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

.PHONY: all clean deps pack

all:
ifeq ($(wildcard build/.),)
	mkdir -p build
	cd build; \
	cmake -DDEPS_PATH=../deps/install ..
endif
	$(MAKE) -C build

clean:
ifeq ($(ALL),1)
	rm -rf build deps
else
	$(MAKE) -C build clean
endif

deps:
	./get_deps.sh $(DEPS_FLAGS)

pack:
	@[ ! -z `command -v redis-server` ] || { echo "Cannot find redis-server - aborting."; exit 1; }
	@[ ! -e $(REDIS_ENT_LIB_PATH) ] || { echo "$(REDIS_ENT_LIB_PATH) exists - aborting."; exit 1; }
ifeq ($(wildcard build/pyenv/.),)
	@virtualenv build/pyenv ;\
	. ./build/pyenv/bin/activate ;\
	pip install git+https://github.com/RedisLabs/RAMP
endif
	@set -xe ;\
	. ./build/pyenv/bin/activate ;\
	ln -fs $(PWD)/deps/install/lib/ $(REDIS_ENT_LIB_PATH) ;\
	ramp pack -m $(PWD)/ramp.yml -o "build/redisai.{os}-{architecture}.${PACK_VER}.zip" $(PWD)/build/ramp/redisai.so ;\
	rm /opt/redislabs/lib
	@cd deps/install/lib; \
	tar pczf ../../../build/redisai-dependencies.${PACK_VER}.tgz *.so*
