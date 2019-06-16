
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

BINDIR=$(PWD)/build

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

# in pack: create ramp/redisai.so with RUNPATH set to /opt/redislabs/lib for RLEC compliance
pack:
	@[ ! -z `command -v redis-server` ] || { echo "Cannot find redis-server - aborting."; exit 1; }
	@[ ! -e $(REDIS_ENT_LIB_PATH) ] || { echo "$(REDIS_ENT_LIB_PATH) exists - aborting."; exit 1; }
ifeq ($(wildcard build/pyenv/.),)
	@virtualenv build/pyenv ;\
	. ./build/pyenv/bin/activate ;\
	pip install git+https://github.com/RedisLabs/RAMP
endif
	@echo "Building RAMP file ..."
	@mkdir -p $(BINDIR)/ramp
	@cp -f $(BINDIR)/redisai.so $(BINDIR)/ramp/
	@patchelf --set-rpath $(REDIS_ENT_LIB_PATH) $(BINDIR)/redisai.so
	@set -e ;\
	. ./build/pyenv/bin/activate ;\
	ln -fs $(PWD)/deps/install/lib/ $(REDIS_ENT_LIB_PATH) ;\
	ramp pack -m $(PWD)/ramp.yml -o "build/redisai.{os}-{architecture}.${PACK_VER}.zip" $(BINDIR)/ramp/redisai.so 2>&1 > /dev/null ;\
	rm $(REDIS_ENT_LIB_PATH)
	@echo Done.
	@echo "Building dependencies file redisai-dependencies.${PACK_VER}.tgz ..."
	@cd deps/install/lib; \
	tar pczf ../../../build/redisai-dependencies.${PACK_VER}.tgz *.so*
	@echo Done.

